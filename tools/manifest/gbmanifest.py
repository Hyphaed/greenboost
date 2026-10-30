#!/usr/bin/env python3
"""gbmanifest: manifest-driven plan / apply / verify / uninstall / coverage for the
GreenBoost installer (prototype, QC2 / SP41-13). Round 2: models line edits.

One JSON manifest lists everything the installer touches. `generate` derives it
from greenboost_setup.sh by static reading only (nothing is executed).
`plan`, `apply`, `verify`, `uninstall`, `coverage` all read that same manifest.

Entry kinds (version 2):
  file         whole file written by the installer (static content, repo source,
               or opaque: content depends on a runtime value). `transforms` holds
               static `sed -i` edits the installer applies to a file it also
               installs (folded in, so the desired bytes are exact).
  line_edit    in-place edit of a file the installer does NOT own (sed -i, `>>`):
               modelled as ordered ops; first-only backup; uninstall restores it.
  dir          directory the installer creates (install -d, mkdir -p).
  group, user, unit_enable, kernel_module   as in round 1 (all modelled as plain
               files under --root, no sudo, no systemctl, no modprobe).

Safety: apply and uninstall refuse root "/" in this prototype. Use --root DIR.
"""
import argparse, difflib, fnmatch, hashlib, json, os, re, sys

VERSION = 2
STATE_REL = "var/lib/greenboost/manifest-state.json"
BACKUP_REL = "var/lib/greenboost/manifest-backups"
MODULES_REL = "run/gbm-modules"
PH_MODULE = "@@MODULE_DIR@@"          # installer's $MODULE_DIR (the checkout), a plan-time parameter
REMOVAL_FUNC = re.compile(r"(purge|uninstall|restore|revert|remove|cleanup|abort_recover)", re.I)
SCRATCH_PREFIXES = ("/tmp/", "/run/", "/proc/", "/sys/", "/dev/")


def rd(p, mode="r"):
    with open(p, mode) as fh:
        return fh.read()


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Unsupported(Exception):
    pass


class Incomplete(Exception):
    pass


class _Scratch:
    def __repr__(self):
        return "<scratch>"


SCR = _Scratch()      # value of a variable that holds a mktemp name
UNSET = object()


# ------------------------------------------------------- shell tokenizer
OP_RE = re.compile(r"(\d*>>|\d*>&\d+|&>>?|\d*>|<<<|<<-?|<|&&|\|\||\||;|\(|\)|\{\s|\})")


def _skip_cmdsub(s, i):
    """s[i] is the char after '$(' ; return index after the matching ')'."""
    depth, n = 1, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "'":
            j = s.find("'", i + 1)
            if j < 0:
                raise Incomplete()
            i = j + 1
            continue
        if c == '"':
            i = _skip_dq(s, i + 1)
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise Incomplete()


def _skip_dq(s, i):
    """s[i] is the char after an opening '"'; return index after the closing quote."""
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        if c == "$" and i + 1 < n and s[i + 1] == "(":
            i = _skip_cmdsub(s, i + 2)
            continue
        if c == "`":
            j = s.find("`", i + 1)
            if j < 0:
                raise Incomplete()
            i = j + 1
            continue
        i += 1
    raise Incomplete()


def tokenize(s):
    """-> [("op", text) | ("w", [(kind, text), ...])]; kind is sq | dq | bare."""
    toks, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\n":
            i += 1
            continue
        if c == "#":
            break
        m = OP_RE.match(s, i)
        if m and (c in "<>|;&(){" or c.isdigit() and re.match(r"\d+>", s[i:])):
            toks.append(("op", m.group(1).strip() if m.group(1).startswith("{") else m.group(1)))
            i = m.end()
            continue
        segs = []

        def put(kind, text):
            if segs and segs[-1][0] == kind:
                segs[-1] = (kind, segs[-1][1] + text)
            else:
                segs.append((kind, text))
        while i < n and s[i] not in " \t\n":
            c = s[i]
            if c in "<>|;&()" and not (c == "&" and False):
                break
            if c == "'":
                j = s.find("'", i + 1)
                if j < 0:
                    raise Incomplete()
                put("sq", s[i + 1:j])
                i = j + 1
            elif c == '"':
                j = _skip_dq(s, i + 1)
                put("dq", s[i + 1:j - 1])
                i = j
            elif c == "\\":
                if i + 1 < n:
                    if s[i + 1] != "\n":
                        put("sq", s[i + 1])
                    i += 2
                else:
                    i += 1
            elif c == "$" and i + 1 < n and s[i + 1] == "(":
                j = _skip_cmdsub(s, i + 2)
                put("bare", s[i:j])
                i = j
            elif c == "`":
                j = s.find("`", i + 1)
                if j < 0:
                    raise Incomplete()
                put("bare", s[i:j + 1])
                i = j + 1
            elif c == "$" and i + 1 < n and s[i + 1] == "{":
                j = s.find("}", i + 2)
                if j < 0:
                    raise Incomplete()
                put("bare", s[i:j + 1])
                i = j + 1
            else:
                put("bare", c)
                i += 1
        if segs:
            toks.append(("w", segs))
        elif i < n and s[i] in "<>|;&()":
            toks.append(("op", s[i]))
            i += 1
        else:
            i += 1
    return toks


# --------------------------------------------------------------- scopes
class Scope:
    def __init__(self, glob):
        self.g = glob
        self.f = {}

    def get(self, k):
        if k in self.f:
            return self.f[k]
        return self.g.get(k, UNSET)


VAR_TOK = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}|\$(\w+)|\$\$|\$\(|\$\{|`")


def expand_text(text, scope, dq, missing, defaulted):
    """Expand variables in a word segment. Unknown / dynamic parts stay verbatim and
    are recorded in `missing`."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == "\\" and i + 1 < n:
            nx = text[i + 1]
            if dq:
                if nx in '"\\$`':
                    out.append(nx)
                elif nx == "\n":
                    pass
                else:
                    out.append("\\" + nx)
            else:
                out.append(nx)
            i += 2
            continue
        if c == "$" or c == "`":
            m = VAR_TOK.match(text, i)
            if not m:
                out.append(c)
                i += 1
                continue
            if m.group(0) in ("$$", "`", "$(", "${"):
                # command substitution / pid / complex expansion: dynamic
                if m.group(0) == "$(":
                    try:
                        j = _skip_cmdsub(text, m.end())
                    except Incomplete:
                        j = n
                    missing.add("$(...)")
                    out.append(text[i:j])
                    i = j
                    continue
                if m.group(0) == "`":
                    j = text.find("`", i + 1)
                    j = n if j < 0 else j + 1
                    missing.add("`...`")
                    out.append(text[i:j])
                    i = j
                    continue
                j = text.find("}", i)
                j = n if j < 0 else j + 1
                missing.add(text[i:j])
                out.append(text[i:j])
                i = j
                continue
            name = m.group(1) or m.group(3)
            default = m.group(2)
            v = scope.get(name)
            if v is UNSET or v is None or v is SCR or v == "":
                if default is not None and (v is UNSET or v == ""):
                    sub = expand_text(default, scope, dq, missing, defaulted)
                    defaulted.add(name)
                    out.append(sub)
                else:
                    missing.add(name)
                    out.append(m.group(0))
            else:
                out.append(v)
            i = m.end()
            continue
        out.append(c)
        i += 1
    return "".join(out)


def word_text(segs, scope):
    """-> (text, missing:set, defaulted:set, quoted_any:bool)"""
    missing, defaulted, parts = set(), set(), []
    for kind, text in segs:
        if kind == "sq":
            parts.append(text)
        else:
            parts.append(expand_text(text, scope, kind == "dq", missing, defaulted))
    return "".join(parts), missing, defaulted


def has_glob(segs):
    return any(k == "bare" and re.search(r"[*?\[]", t) for k, t in segs)


# ------------------------------------------------------------- sed model
def bre(p):
    """GNU sed basic regex (the subset the installer uses) -> Python re."""
    out, i, n = [], 0, len(p)
    while i < n:
        c = p[i]
        if c == "[":
            j = i + 1
            if j < n and p[j] == "^":
                j += 1
            if j < n and p[j] == "]":
                j += 1
            while j < n and p[j] != "]":
                if p[j] == "[" and j + 1 < n and p[j + 1] == ":":
                    k = p.find(":]", j)
                    j = k + 2 if k > 0 else j + 1
                else:
                    j += 1
            cls = p[i:j + 1]
            cls = cls.replace("[:space:]", r"\s").replace("[:alpha:]", "a-zA-Z").replace("[:digit:]", "0-9")
            out.append(cls)
            i = j + 1
        elif c == "\\" and i + 1 < n:
            nx = p[i + 1]
            if nx in "(){}|+?":
                out.append(nx)
            elif nx in "<>":
                raise Unsupported("word boundary \\%s" % nx)
            elif nx == "/":
                out.append("/")
            else:
                out.append("\\" + nx)
            i += 2
        elif c in "(){}|+?":
            out.append("\\" + c)
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def parse_sed(script):
    """-> [(kind, addr_regex|None, arg)] ; kinds d, a, i, s. Raises Unsupported."""
    stmts, i, n = [], 0, len(script)
    while i < n:
        while i < n and script[i] in " \t\n;":
            i += 1
        if i >= n:
            break
        addr = None
        if script[i] == "/" or (script[i] == "\\" and i + 1 < n):
            d = "/" if script[i] == "/" else script[i + 1]
            k = i + 1 if script[i] == "/" else i + 2
            buf = []
            while k < n and script[k] != d:
                if script[k] == "\\" and k + 1 < n:
                    buf.append(d if script[k + 1] == d else script[k:k + 2])
                    k += 2
                    continue
                buf.append(script[k])
                k += 1
            if k >= n:
                raise Unsupported("unterminated address")
            addr = bre("".join(buf))
            re.compile(addr)
            i = k + 1
            while i < n and script[i] in " \t":
                i += 1
        if i >= n:
            raise Unsupported("address without command")
        cmd = script[i]
        if cmd == "d":
            stmts.append(("d", addr, None))
            i += 1
        elif cmd in "ai":
            i += 1
            if i < n and script[i] == "\\":
                i += 1
                if i < n and script[i] == "\n":
                    i += 1
            else:
                while i < n and script[i] in " \t":
                    i += 1
            j = i
            while j < n and script[j] != "\n":
                if script[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                j += 1
            text = script[i:j].replace("\\\\", "\x00").replace("\\n", "\n").replace("\x00", "\\")
            stmts.append((cmd, addr, text))
            i = j
        elif cmd == "s":
            dl = script[i + 1]
            parts, k, buf = [], i + 2, []
            while k < n and len(parts) < 2:
                if script[k] == "\\" and k + 1 < n:
                    buf.append(dl if script[k + 1] == dl else script[k:k + 2])
                    k += 2
                    continue
                if script[k] == dl:
                    parts.append("".join(buf))
                    buf = []
                    k += 1
                    continue
                buf.append(script[k])
                k += 1
            if len(parts) < 2:
                raise Unsupported("bad s command")
            flags = ""
            while k < n and script[k] not in " \t\n;":
                flags += script[k]
                k += 1
            if re.search(r"(?<!\\)&|\\[0-9]", parts[1]) or flags.replace("g", ""):
                raise Unsupported("s with & / backreference / flags")
            stmts.append(("s", bre(parts[0]), (parts[1], "g" in flags)))
            i = k
        else:
            raise Unsupported("sed command %r" % cmd)
    return stmts


def apply_sed(text, stmts):
    lines = text.split("\n")
    trailing = text.endswith("\n")
    if trailing:
        lines.pop()
    out = []
    for ln in lines:
        ps, pre, post, dele = ln, [], [], False
        for kind, addr, arg in stmts:
            if kind != "s" and addr is not None and not re.search(addr, ps):
                continue
            if kind == "d":
                dele = True
                break
            if kind == "a":
                post.append(arg)
            elif kind == "i":
                pre.append(arg)
            elif kind == "s":
                ps = re.sub(addr, arg[0].replace("\\", "\\\\"), ps, count=0 if arg[1] else 1)
        out.extend(pre)
        if not dele:
            out.append(ps)
        out.extend(post)
    return "\n".join(out) + ("\n" if (trailing and out) else "")


def apply_ops(text, ops):
    for op in ops:
        guard = op.get("unless_re")
        if guard and re.search(bre(guard), text, re.M):
            continue
        if op["op"] == "sed":
            text = apply_sed(text, parse_sed(op["script"]))
        elif op["op"] == "append_lines":
            have = set(text.split("\n"))
            add = [l for l in op["lines"] if l not in have]
            if add:
                if text and not text.endswith("\n"):
                    text += "\n"
                text += "\n".join(add) + "\n"
    return text


# ------------------------------------------------------------- generator
FUNC_RE = re.compile(r"^(\w+)\(\)\s*\{")
GLOBAL_VAR_RE = re.compile(r'^([A-Z_][A-Z0-9_]*)="([^"`]*)"\s*(#.*)?$')
HERE_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_]\w*)\1")
ASSIGN_RE = re.compile(r"^(?:local\s+(?:-\w+\s+)*)?([A-Za-z_]\w*)=(.*)$")
DIRNAME_SELF = '$(dirname "$(realpath "${BASH_SOURCE[0]}")")'
SYSTEM_ROOTS = ("/etc/", "/usr/", "/var/", "/opt/", "/lib/", "/boot/")
SIDE_EFFECT_CMDS = ("systemctl", "udevadm", "ldconfig", "update-grub", "grub-mkconfig", "dracut",
                    "update-initramfs", "mkinitcpio", "depmod", "sysctl", "swapon", "swapoff", "mkswap",
                    "systemd-tmpfiles", "ufw", "firewall-cmd", "iptables", "apparmor_parser", "dkms",
                    "apt-get", "dnf", "zypper", "pacman", "dpkg", "pip", "usermod", "setfacl", "visudo")


class Gen:
    def __init__(self, repo):
        self.repo = repo
        self.entries, self.unresolved, self.removals = [], [], []
        self.remote, self.scratch, self.perms = [], [], []
        self.commands = {}
        self.seen = {}
        self.func = "<top>"
        self.last_pipe = None      # body of a `cat <<X` with no redirect (feeds `| gb_managed_write`)
        self.tmpw = {}             # scratch var -> pending file draft
        self.last_cp = None
        self.unloads = set()
        self.var_echo_writes = 0
        self.lineno = 0
        self.lines = []

    # -- helpers
    def step(self):
        return "%s:%d" % (self.func, self.lineno)

    def removal_ctx(self):
        return bool(REMOVAL_FUNC.search(self.func))

    def add(self, e):
        key = (e["kind"], e.get("path") or e.get("name"))
        if key in self.seen:
            prev = self.seen[key]
            prev["steps"].append(self.step())
            if e["kind"] == "file" and e.get("path") and (prev.get("content"), prev.get("source")) != (e.get("content"), e.get("source")) \
                    and (e.get("content") is not None or e.get("source")) and (prev.get("content") is not None or prev.get("source")):
                prev.setdefault("conflicting_writers", []).append(self.step())
                prev["opaque"] = True
                prev["why_opaque"] = "written with different content by more than one function (last writer wins by command order): %s" % \
                    ", ".join([prev["steps"][0]] + prev["conflicting_writers"])
            return prev
        e["steps"] = [self.step()]
        e["id"] = "%s:%s" % (e["kind"], key[1])
        self.seen[key] = e
        self.entries.append(e)
        return e

    def unres(self, what, target, why, category):
        self.unresolved.append({"what": what, "target": target, "step": self.step(), "why": why, "category": category})

    def path_of(self, segs):
        """resolve a word to an absolute literal path -> (path|None, reason|None)"""
        text, missing, _ = word_text(segs, self.scope)
        if has_glob(segs) or re.search(r"[*?]", text):
            return None, "glob"
        if missing:
            return None, "var:" + ",".join(sorted(missing))
        if not text.startswith("/"):
            return None, "relative"
        return text, None

    def source_of(self, segs):
        text, missing, _ = word_text(segs, self.scope)
        if PH_MODULE in text:
            rel = text.replace(PH_MODULE, "").lstrip("/")
        elif text.startswith(self.repo + "/"):
            rel = text[len(self.repo) + 1:]
        else:
            return None
        if re.search(r"[$`]", rel) or ".." in rel.split("/") or not rel:
            return None
        return rel if os.path.isfile(os.path.join(self.repo, rel)) else None

    def vars_scan(self, lines):
        raw = {"MODULE_DIR": PH_MODULE}
        for ln in lines:
            m = GLOBAL_VAR_RE.match(ln)
            if m:
                raw.setdefault(m.group(1), m.group(2).replace("${MODULE_DIR}", PH_MODULE).replace("$MODULE_DIR", PH_MODULE))
        # second round: expand references between globals
        sc = Scope(raw)
        for k in list(raw):
            miss = set()
            raw[k] = expand_text(raw[k], sc, True, miss, set())
        return raw

    # -- logical lines
    def logical(self, i):
        """-> (text, next_index). Joins backslash continuations and unterminated quotes."""
        text = self.lines[i][0]
        j = i + 1
        for _ in range(80):
            cont = text.rstrip().endswith("\\") and not text.rstrip().endswith("\\\\")
            if cont:
                if j >= len(self.lines):
                    break
                text = text.rstrip()[:-1] + " " + self.lines[j][0].strip()
                j += 1
                continue
            if HERE_RE.search(text):
                return text, j          # the body follows; never merge it into the statement
            try:
                tokenize(text)
                return text, j
            except Incomplete:
                if j >= len(self.lines):
                    break
                text = text + "\n" + self.lines[j][0]
                j += 1
        return self.lines[i][0], i + 1

    # -- main walk
    def run(self, lines):
        saved = (self.lines,)
        self.lines = lines
        i = 0
        while i < len(lines):
            text, ln = lines[i]
            self.lineno = ln
            st = text.strip()
            fm = FUNC_RE.match(text)
            if fm:
                self.func = fm.group(1)
                self.scope.f = {}
                self.tmpw = {}
                i += 1
                continue
            if not st or st.startswith("#"):
                i += 1
                continue
            if re.match(r"^\s*(?:local\s+(?:-\w+\s+)*)?\w+=\(\s*(#.*)?$", text):
                j = i + 1
                while j < len(lines) and not self.arr_end(lines[j][0]):
                    j += 1
                i = j + 1
                continue
            if st == "{":
                i = self.group_block(lines, i)
                continue
            ltext, j = self.logical(i)
            hm = HERE_RE.search(ltext)
            if hm:
                delim, quoted = hm.group(2), hm.group(1) != ""
                jj, buf = j, []
                while jj < len(lines) and lines[jj][0].strip() != delim and lines[jj][0].lstrip("\t") != delim:
                    buf.append(lines[jj][0])
                    jj += 1
                body = "\n".join(buf) + ("\n" if buf else "")
                self.heredoc(ltext, body, quoted)
                i = jj + 1
                continue
            fr = re.match(r"^\s*for\s+(\w+)\s+in\s+", text)
            if fr:
                i = self.for_loop(ltext, j, lines, i)
                continue
            try:
                self.statement(ltext)
            except (Unsupported, Incomplete) as ex:
                self.unres("statement", ltext.strip()[:80], str(ex) or "unbalanced quotes", "unsupported")
            i = j
        self.lines = saved[0]

    def for_loop(self, header, j, lines, i0):
        single = re.search(r";\s*done\s*$", header)
        if single:
            header = re.sub(r";\s*done\s*$", "", header)
        m = re.match(r"^\s*for\s+(\w+)\s+in\s+(.*?)\s*;?\s*do\b(.*)$", header, re.S)
        if not m:
            m2 = re.match(r"^\s*for\s+(\w+)\s+in\s+(.*)$", header, re.S)
            if not m2:
                return j
            var, words, rest = m2.group(1), m2.group(2).rstrip("; \t"), ""
            if j < len(lines) and lines[j][0].strip() == "do":
                j += 1
        else:
            var, words, rest = m.group(1), m.group(2), m.group(3)
        depth, k = 1, j
        if single:
            k = j - 1
            body_override = [(rest, lines[i0][1])] if rest.strip() else []
        while k < len(lines) and not single:
            s = lines[k][0].strip()
            if re.match(r"^(for|while|until)\b", s):
                depth += 1
            elif re.match(r"^done\b", s):
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = ([(rest, lines[i0][1])] if rest.strip() else []) + ([] if single else lines[j:k])
        values = None
        try:
            toks = tokenize(words)
            vals = []
            for kind, segs in toks:
                if kind != "w":
                    raise Unsupported("op in word list")
                raw = "".join(t for _, t in segs)
                am = re.fullmatch(r"\$\{(\w+)\[@\]\}", raw)
                if am and am.group(1) in self.arrays:
                    vals.extend(self.arrays[am.group(1)])
                    if am.group(1) in self.dyn_arrays:
                        self.unres("loop", am.group(1), "array is extended at run time (+=); only its static defaults are modelled", "dynamic-set")
                    continue
                text, missing, _ = word_text(segs, self.scope)
                if missing:
                    raise Unsupported("dynamic")
                if has_glob(segs) and not (self.removal_ctx() and text.startswith("/")):
                    raise Unsupported("glob")
                vals.append(text)
            values = vals
        except (Unsupported, Incomplete):
            values = None
        saved = self.scope.f.get(var, UNSET)
        if values is None:
            self.scope.f[var] = None
            self.run(body)
        else:
            for v in values:
                self.scope.f[var] = v
                self.run(body)
        if saved is UNSET:
            self.scope.f.pop(var, None)
        else:
            self.scope.f[var] = saved
        return j if single else k + 1

    def wt(self, segs):
        return word_text(segs, self.scope)[0]

    def group_block(self, lines, i):
        """`{ ...echo/cat... } >> file` or `{ ... } | gb_managed_write file`: one composed write."""
        text0 = lines[i][0]
        ind = len(text0) - len(text0.lstrip())
        k = i + 1
        while k < len(lines):
            t = lines[k][0]
            if t.strip().startswith("}") and len(t) - len(t.lstrip()) == ind:
                break
            k += 1
        if k >= len(lines):
            return i + 1
        parts, cond, miss, j = [], False, set(), i + 1
        while j < k:
            t, ln = lines[j]
            self.lineno = ln
            st = t.strip()
            hm = HERE_RE.search(t)
            if hm and st.startswith("cat"):
                delim, quoted = hm.group(2), hm.group(1) != ""
                jj, buf = j + 1, []
                while jj < k and lines[jj][0].strip() != delim:
                    buf.append(lines[jj][0])
                    jj += 1
                body = "\n".join(buf) + "\n"
                txt, m, d = self.render_heredoc(body, quoted)
                parts.append(txt)
                miss |= m | d
                j = jj + 1
                continue
            if re.match(r"^(echo|printf)\b", st):
                try:
                    toks = [x for x in tokenize(st) if x[0] == "w"]
                    argv = [x[1] for x in toks[1:]]
                    if st.startswith("echo"):
                        txt, m, d = " ".join(word_text(a, self.scope)[0] for a in argv) + "\n", set(), set()
                        for a in argv:
                            m |= word_text(a, self.scope)[1]
                    else:
                        fmt, m, d = word_text(argv[0], self.scope)
                        args = []
                        for a in argv[1:]:
                            tt, mm, dd = word_text(a, self.scope)
                            m |= mm
                            args.append(tt)
                        txt = self.printf(fmt, args)
                    parts.append(txt)
                    miss |= m
                except Incomplete:
                    cond = True
            elif st and not st.startswith("#"):
                cond = cond or bool(re.search(r"\b(echo|printf)\b", st))
            j += 1
        content = "".join(parts)
        tail = lines[k][0].strip()[1:]
        self.lineno = lines[k][1]
        try:
            toks = tokenize(tail)
        except Incomplete:
            return k + 1
        ops = [x for x in toks if x[0] == "op"]
        if toks and toks[0][0] == "op" and toks[0][1] in (">", ">>") and len(toks) > 1 and toks[1][0] == "w":
            self.emit_write(toks[1][1], content, miss | ({"<conditional>"} if cond else set()), set(), toks[0][1] == ">>", "group")
        elif toks and toks[0] == ("op", "|") and len(toks) > 2 and toks[1][0] == "w" and self.wt(toks[1][1]) == "gb_managed_write":
            self.last_pipe = {"text": content, "miss": miss, "line": self.lineno}
            self.cmd_gb_managed_write([t[1] for t in toks[2:] if t[0] == "w"], [], tail)
            if cond or miss:
                e = self.seen.get(("file", self.path_of([t[1] for t in toks[2:] if t[0] == "w"][0])[0]))
                if e is not None:
                    e["runtime_vars"] = sorted(miss | ({"<conditional lines>"} if cond else set()))
        return k + 1

    # -- statements
    def split_stmts(self, toks):
        segs, cur = [], []
        for t in toks:
            if t[0] == "op" and t[1] in ("&&", "||", ";", "|"):
                segs.append((cur, t[1]))
                cur = []
            else:
                cur.append(t)
        segs.append((cur, None))
        return segs

    def statement(self, text):
        # array definition `name=(` ... is handled by pre-scan; skip bodies
        toks = tokenize(text)
        for cur, sep in self.split_stmts(toks):
            cur = [t for t in cur if not (t[0] == "op" and t[1] in ("(", ")", "{", "}"))]
            while cur and cur[0][0] == "w" and "".join(x for _, x in cur[0][1]) in ("then", "do", "else", "if", "elif", "!", "{", "local", "export", "readonly") and len(cur) > 1:
                kw = "".join(x for _, x in cur[0][1])
                if kw in ("local", "export", "readonly"):
                    break
                cur = cur[1:]
            if not cur:
                continue
            self.command(cur, text, sep)

    def command(self, cur, text, sep):
        words = [t for t in cur if t[0] == "w"]
        if not words:
            return
        first = "".join(x for _, x in words[0][1])
        if first in ("local", "export", "readonly", "declare", "typeset"):
            for t in words[1:]:
                raw = "".join(x for _, x in t[1])
                if t[1][0][0] == "bare" and "=" in raw and re.match(r"\w+=", raw):
                    self.assign_word(t[1])
                elif re.fullmatch(r"\w+", raw):
                    self.scope.f.setdefault(raw, None)
            return
        if re.match(r"^[A-Za-z_]\w*=", "".join(x for _, x in words[0][1])) and len(words) == 1:
            self.assign_word(words[0][1])
            return
        if first == "sudo":
            cur = cur[1:]
            words = words[1:]
            if not words:
                return
            first = "".join(x for _, x in words[0][1])
        # redirects
        redirs = []
        args = []
        k = 0
        while k < len(cur):
            t = cur[k]
            if t[0] == "op" and re.match(r"\d*>>?$|&>>?$", t[1]) and k + 1 < len(cur) and cur[k + 1][0] == "w":
                redirs.append((t[1], cur[k + 1][1]))
                k += 2
                continue
            if t[0] == "op" and (re.match(r"\d*>&\d+$", t[1]) or t[1].startswith("<")):
                k += 2 if t[1] in ("<", "<<<") else 1
                continue
            if t[0] == "w":
                args.append(t[1])
            k += 1
        name = "".join(x for _, x in args[0]) if args else first
        argv = args[1:]
        h = getattr(self, "cmd_" + re.sub(r"\W", "_", name), None)
        if name in SIDE_EFFECT_CMDS or name == "modprobe" or name == "rmmod":
            self.commands.setdefault(name, []).append(self.step())
        if h:
            h(argv, redirs, text)
        elif name in ("echo", "printf") or redirs:
            self.write_cmd(name, argv, redirs)

    # -- assignment
    def assign_word(self, segs):
        raw0 = segs[0][1]
        m = re.match(r"^([A-Za-z_]\w*)=", raw0)
        if not m:
            return
        name = m.group(1)
        val_segs = [(segs[0][0], raw0[m.end():])] + segs[1:]
        txt_raw = "".join(t for _, t in val_segs)
        if "mktemp" in txt_raw:
            self.scope.f[name] = SCR
            return
        val_segs = [(k, t.replace(DIRNAME_SELF, "${MODULE_DIR}")) for k, t in val_segs]
        text, missing, _ = word_text(val_segs, self.scope)
        self.scope.f[name] = None if missing else text

    @staticmethod
    def arr_end(t):
        t = t.strip()
        return t == ")" or (t.endswith(")") and "(" not in t and not t.startswith("#"))

    def pre_arrays(self, lines):
        self.arrays = {}
        i = 0
        while i < len(lines):
            m = re.match(r"^\s*(?:local\s+(?:-\w+\s+)*)?(\w+)=\(\s*(#.*)?$", lines[i][0])
            if m:
                items, j = [], i + 1
                while j < len(lines):
                    s = lines[j][0].strip()
                    last = self.arr_end(s)
                    if last and s != ")":
                        s = s[:-1].rstrip()
                    if s == ")":
                        break
                    if s and not s.startswith("#"):
                        try:
                            toks = [t for t in tokenize(s) if t[0] == "w"]
                        except Incomplete:
                            toks = []
                        for kind, segs in toks:
                            items.append("".join(x for _, x in segs))
                    if last:
                        break
                    j += 1
                self.arrays[m.group(1)] = items
                i = j
            i += 1
        self.dyn_arrays = {n for n in self.arrays if re.search(r"\b%s\+=\(" % re.escape(n), "\n".join(l[0] for l in lines))}

    # -- commands
    def cmd_install(self, argv, redirs, text):
        mode, is_dir, rest, k = None, False, [], 0
        while k < len(argv):
            a = "".join(x for _, x in argv[k])
            if a == "-d":
                is_dir = True
            elif a == "-m" and k + 1 < len(argv):
                mode = "".join(x for _, x in argv[k + 1])
                k += 1
            elif a.startswith("-m") and len(a) > 2:
                mode = a[2:]
            elif a in ("-o", "-g", "-t") and k + 1 < len(argv):
                k += 1
            elif a.startswith("-"):
                pass
            else:
                rest.append(argv[k])
            k += 1
        if self.removal_ctx():
            return
        if is_dir:
            for segs in rest:
                p, why = self.path_of(segs)
                if p:
                    self.add({"kind": "dir", "path": p, "mode": mode or "755"})
            return
        if len(rest) < 2:
            return
        if len(rest) != 2:
            self.unres("install", self.step(), "install with %d operands" % len(rest), "unsupported")
            return
        self.install_file(rest[0], rest[1], mode or "755")

    def cmd_gb_atomic_install(self, argv, redirs, text):
        if len(argv) >= 2:
            self.install_file(argv[0], argv[1], ("".join(x for _, x in argv[2]) if len(argv) > 2 else "755"))

    def install_file(self, src_segs, dest_segs, mode):
        dest, why = self.path_of(dest_segs)
        if not dest:
            tgt = "".join(x for _, x in dest_segs)
            self.unres("install", tgt, "destination %s" % why, "runtime-path" if why and why.startswith("var:") else "dynamic-set")
            return
        if dest.endswith("/"):
            return
        s = self.source_of(src_segs)
        e = {"kind": "file", "path": dest, "mode": mode, "source": s, "content": None, "opaque": s is None,
             "why_opaque": None if s else "source is not a repo file known statically (%s)" % "".join(x for _, x in src_segs)}
        if s and s.endswith((".so", ".ko")) or s in ("greenboost-netd", "greenboost-ebpf-trace"):
            e["source_kind"] = "build-output"
        self.add(e)

    def cmd_cp(self, argv, redirs, text):
        pos = [a for a in argv if not "".join(x for _, x in a).startswith("-")]
        if len(pos) != 2:
            return
        self.last_cp = (self.raw(pos[0]), self.raw(pos[1]), pos[0], pos[1])
        if self.removal_ctx():
            return
        dest, why = self.path_of(pos[1])
        if dest and dest.startswith(SYSTEM_ROOTS) and not os.path.basename(dest).startswith(".") and "/.gb" not in dest:
            s = self.source_of(pos[0])
            if s:
                self.install_file(pos[0], pos[1], "755" if s.endswith((".so", ".sh")) else "644")

    def cmd_mv(self, argv, redirs, text):
        pos = [a for a in argv if not "".join(x for _, x in a).startswith("-")]
        if len(pos) != 2 or self.removal_ctx():
            return
        srcraw = self.raw(pos[0])
        dest, why = self.path_of(pos[1])
        if self.last_cp and self.last_cp[1] == srcraw and dest and dest.startswith(SYSTEM_ROOTS):
            self.install_file(self.last_cp[2], pos[1], "755")
            return
        var = re.fullmatch(r"\$\{?(\w+)\}?", srcraw)
        if var and var.group(1) in self.tmpw and dest:
            d = self.tmpw.pop(var.group(1))
            d["path"] = dest
            self.add(d)

    def cmd_chmod(self, argv, redirs, text):
        argv = [a for a in argv if not self.wt(a).startswith("-R")]
        if len(argv) < 2:
            return
        mode = self.wt(argv[0])
        tgt = argv[-1]
        raw = self.raw(tgt)
        var = re.fullmatch(r"\$\{?(\w+)\}?", raw)
        if var and var.group(1) in self.tmpw:
            if re.fullmatch(r"[0-7]{3,4}", mode):
                self.tmpw[var.group(1)]["mode"] = mode.lstrip("0") or "0"
            return
        p, why = self.path_of(tgt)
        if not p:
            return
        for e in self.entries:
            if e["kind"] == "file" and e["path"] == p:
                if re.fullmatch(r"[0-7]{3,4}", mode):
                    e["mode"] = mode.lstrip("0") or "0"
                elif mode == "+x" and e["mode"] == "644":
                    e["mode"] = "755"
                return
        self.perms.append({"op": "chmod", "path": p, "mode": mode, "step": self.step()})

    def cmd_chown(self, argv, redirs, text):
        if not argv or self.removal_ctx():
            return
        argv = [a for a in argv if not self.wt(a).startswith("-")]
        if len(argv) < 2:
            return
        tgt = argv[-1]
        p, why = self.path_of(tgt)
        who = self.wt(argv[0])
        self.perms.append({"op": "chown", "path": p or self.raw(tgt), "owner": who, "step": self.step()})

    def cmd_mkdir(self, argv, redirs, text):
        if self.removal_ctx():
            return
        for segs in argv:
            if "".join(x for _, x in segs).startswith("-"):
                continue
            p, why = self.path_of(segs)
            if p and p.startswith(SYSTEM_ROOTS) and not has_glob(segs):
                self.add({"kind": "dir", "path": p, "mode": "755"})

    def cmd_rm(self, argv, redirs, text):
        rec = False
        for segs in argv:
            a = "".join(x for _, x in segs)
            if a.startswith("-"):
                rec = rec or "r" in a or "R" in a
                continue
            t, missing, _ = word_text(segs, self.scope)
            if missing or not t.startswith("/"):
                continue
            self.removals.append({"target": t, "recursive": rec, "step": self.step(), "removal_ctx": self.removal_ctx()})

    def cmd_systemctl(self, argv, redirs, text):
        words = [self.wt(a) for a in argv]
        words = [w for w in words if not w.startswith("-")]
        if not words:
            return
        verb, units = words[0], words[1:]
        for u in units:
            if verb == "enable" and re.search(r"\.(service|timer|socket|path)$", u) and not self.removal_ctx():
                self.add({"kind": "unit_enable", "name": u, "wanted_by": None})
            elif verb in ("disable", "mask"):
                self.removals.append({"target": "unit:" + u, "recursive": False, "step": self.step(), "removal_ctx": self.removal_ctx()})

    def cmd_groupadd(self, argv, redirs, text):
        names = [self.wt(a) for a in argv if not self.wt(a).startswith("-")]
        if names and not self.removal_ctx() and "$" not in names[-1]:
            self.add({"kind": "group", "name": names[-1]})

    def cmd_groupdel(self, argv, redirs, text):
        names = [self.wt(a) for a in argv if not self.wt(a).startswith("-")]
        if names and "$" not in names[-1]:
            self.removals.append({"target": "group:" + names[-1], "recursive": False, "step": self.step(), "removal_ctx": self.removal_ctx()})

    def cmd_useradd(self, argv, redirs, text):
        names = [self.wt(a) for a in argv if not self.wt(a).startswith("-")]
        if names and not self.removal_ctx() and "$" not in names[-1]:
            self.add({"kind": "user", "name": names[-1]})

    def cmd_usermod(self, argv, redirs, text):
        self.unres("group-membership", "".join(x for a in argv for _, x in a), "adds the invoking user to a group: the user is known only at run time", "runtime-user")

    def cmd_gpasswd(self, argv, redirs, text):
        self.cmd_usermod(argv, redirs, text)

    def cmd_modprobe(self, argv, redirs, text):
        words = [self.wt(a) for a in argv]
        if "-r" in words:
            self.unloads.update(w for w in words if not w.startswith("-"))
            return
        names = [w for w in words if not w.startswith("-") and re.fullmatch(r"[\w-]+", w)]
        if names and not self.removal_ctx():
            self.add({"kind": "kernel_module", "name": names[0]})

    def cmd_rmmod(self, argv, redirs, text):
        self.unloads.update(self.wt(a) for a in argv)

    # -- sed -i
    def cmd_sed(self, argv, redirs, text):
        words = ["".join(x for _, x in a) for a in argv]
        if not any(re.fullmatch(r"-[a-zA-Z]*i\S*|--in-place", w) for w in words):
            return
        script_segs, files, k, got = None, [], 0, False
        while k < len(argv):
            w = words[k]
            if w.startswith("-") and not got:
                if w in ("-e",) and k + 1 < len(argv):
                    script_segs = argv[k + 1]
                    got = True
                    k += 1
                k += 1
                continue
            if script_segs is None:
                script_segs = argv[k]
            else:
                files.append(argv[k])
            k += 1
        if script_segs is None or not files:
            return
        stext, smiss, _ = word_text(script_segs, self.scope)
        for fsegs in files:
            path, why = self.path_of(fsegs)
            tgt = path or self.raw(fsegs)
            if self.removal_ctx():
                self.removals.append({"target": tgt, "recursive": False, "step": self.step(), "removal_ctx": True, "sed": True})
                continue
            if not path:
                cat = "runtime-path" if (why or "").startswith("var:") else "dynamic-set"
                if why == "glob":
                    cat = "dynamic-set"
                self.unres("sed-edit", tgt, "file %s" % why, cat)
                continue
            if smiss:
                self.unres("sed-edit", path, "script uses runtime value(s): %s" % ", ".join(sorted(smiss)), "runtime-value")
                ex = self.seen.get(("line_edit", path))
                if ex is not None:
                    ex["incomplete"] = "a later edit inserts runtime values (%s); not applied by this tool" % ", ".join(sorted(smiss))
                continue
            try:
                parse_sed(stext)
            except Unsupported as ex:
                self.unres("sed-edit", path, "sed script outside the modelled subset (%s)" % ex, "unsupported-sed")
                continue
            self.add_line_edit(path, {"op": "sed", "script": stext, "unless_re": self.guard()})

    def guard(self):
        """`if ! grep -q 'PAT' file` on the previous 3 source lines guards an edit."""
        idx = None
        for n in range(len(self.lines)):
            if self.lines[n][1] == self.lineno:
                idx = n
                break
        if idx is None:
            return None
        for back in range(1, 7):
            if idx - back < 0:
                break
            m = re.search(r"!\s*grep\s+-q[A-Za-z]*\s+(?:'([^']+)'|\"([^\"$]+)\")", self.lines[idx - back][0])
            if m:
                return m.group(1) or m.group(2)
        return None

    def add_line_edit(self, path, op):
        e = self.add({"kind": "line_edit", "path": path, "ops": [], "only_if_exists": op["op"] == "sed",
                      "revert": "restore_original"})
        op = {k: v for k, v in op.items() if v is not None}
        if op not in e["ops"]:
            e["ops"].append(op)

    # -- echo / printf / redirects
    def raw(self, segs):
        return "".join(t if k != "sq" else t for k, t in segs)

    def write_cmd(self, name, argv, redirs):
        targets = [r for r in redirs if re.match(r"(\d*)>>?$", r[0]) and not r[0].startswith(("2", "1>&"))]
        if not targets:
            return
        op, tsegs = targets[0]
        append = op.endswith(">>")
        texts = []
        miss, dflt = set(), set()
        if name == "echo":
            parts = []
            for a in argv:
                if "".join(x for _, x in a) in ("-e", "-n"):
                    continue
                t, m, d = word_text(a, self.scope)
                miss |= m
                dflt |= d
                parts.append(t)
            content = " ".join(parts) + "\n"
        elif name == "printf":
            if not argv:
                return
            fmt, m, d = word_text(argv[0], self.scope)
            miss |= m
            args = []
            for a in argv[1:]:
                t, m, d = word_text(a, self.scope)
                miss |= m
                dflt |= d
                args.append(t)
            content = self.printf(fmt, args)
        else:
            return
        self.emit_write(tsegs, content, miss, dflt, append, name)

    @staticmethod
    def printf(fmt, args):
        out, ai, i = [], 0, 0
        fmt = fmt.replace("\\n", "\n").replace("\\t", "\t")
        while i < len(fmt):
            if fmt[i] == "%" and i + 1 < len(fmt):
                if fmt[i + 1] == "%":
                    out.append("%")
                    i += 2
                    continue
                m = re.match(r"%[-0-9.]*[sd]", fmt[i:])
                if m:
                    out.append(args[ai] if ai < len(args) else "")
                    ai += 1
                    i += m.end()
                    continue
            out.append(fmt[i])
            i += 1
        return "".join(out)

    def emit_write(self, tsegs, content, miss, dflt, append, how, quoted=True):
        raw_t = self.raw(tsegs)
        vm = re.fullmatch(r"\$\{?(\w+)\}?", raw_t)
        if vm and self.scope.get(vm.group(1)) is SCR:
            draft = {"kind": "file", "path": None, "mode": "644", "content": None if (miss or dflt) else content,
                     "opaque": bool(miss or dflt), "why_opaque": ("depends on runtime value(s): %s" % ", ".join(sorted(miss | dflt))) if (miss or dflt) else None,
                     "template_body": content if (miss or dflt) else None, "via": "tmp+mv"}
            self.tmpw[vm.group(1)] = draft
            self.scratch.append({"var": vm.group(1), "step": self.step()})
            return
        path, why = self.path_of(tsegs)
        if not path:
            if (why or "").startswith("var:") and any(self.scope.get(v) is SCR for v in why[4:].split(",")):
                self.scratch.append({"var": raw_t, "step": self.step()})
                return
            if how in ("echo", "printf"):
                self.var_echo_writes += 1       # `echo x > "$sysfs_file"` style: runtime targets, not installed files
                return
            self.unres("heredoc-append" if append else "heredoc-write", raw_t, "path %s" % (why or "unresolved"),
                       "runtime-path" if (why or "").startswith("var:") else "dynamic-set")
            return
        if path == "/etc/ld.so.preload":
            self.unres("rewrite", path, "rewrites the file without stale GreenBoost lines (runs for any root invocation)", "runtime-value")
            return
        if path.startswith(SCRATCH_PREFIXES) and not path.startswith("/etc"):
            self.scratch.append({"var": path, "step": self.step()})
            return
        if self.removal_ctx():
            return
        if append:
            if miss or dflt:
                self.unres("append", path, "appended text depends on runtime value(s): %s" % ", ".join(sorted(miss | dflt)), "runtime-value")
                return
            lines = [l for l in content.split("\n") if l != ""]
            self.add_line_edit(path, {"op": "append_lines", "lines": lines, "unless_re": self.guard()})
            e = self.seen[("line_edit", path)]
            e["only_if_exists"] = False
            return
        e = {"kind": "file", "path": path, "mode": "644", "content": None if (miss or dflt) else content,
             "opaque": bool(miss or dflt), "why_opaque": ("depends on runtime value(s): %s" % ", ".join(sorted(miss | dflt))) if (miss or dflt) else None}
        if miss or dflt:
            e["template_body"] = content
            e["runtime_vars"] = sorted(miss | dflt)
            e["static_fragments"] = [l for l in content.split("\n") if l.strip() and not l.startswith("#") and not re.search(r"@@|\$\(|\$\{|`", l) and not any(v in l for v in (miss | dflt))]
        prev = self.guard_create(path)
        if prev:
            e["create_only"] = True
        self.add(e)

    def guard_create(self, path):
        for n in range(len(self.lines)):
            if self.lines[n][1] == self.lineno:
                for back in range(1, 3):
                    if n - back >= 0 and re.search(r"!\s*-f\s+", self.lines[n - back][0]):
                        return True
        return False

    def heredoc(self, first_line, body, quoted):
        try:
            toks = tokenize(first_line.split("<<")[0])
        except Incomplete:
            toks = []
        cmd = None
        for t in toks:
            if t[0] == "w":
                cmd = "".join(x for _, x in t[1])
                break
        rest = toks
        if cmd in ("sudo",):
            rest = [t for t in toks if t != toks[0]]
            cmd = "".join(x for _, x in rest[0][1]) if rest and rest[0][0] == "w" else cmd
        redirs, k = [], 0
        while k < len(rest):
            if rest[k][0] == "op" and re.match(r"\d*>>?$", rest[k][1]) and k + 1 < len(rest) and rest[k + 1][0] == "w":
                redirs.append((rest[k][1], rest[k + 1][1]))
                k += 2
                continue
            k += 1
        if re.search(r"\bssh\b", first_line.split("<<")[0]) or re.search(r"\bbash\s+-s\b", first_line):
            self.remote.append({"step": self.step(), "note": "script run on a feeder over ssh; not modelled"})
            return
        if cmd in ("cat", "tee") and redirs:
            text, miss, dflt = self.render_heredoc(body, quoted)
            self.emit_write(redirs[0][1], text, miss, dflt, redirs[0][0].endswith(">>"), "heredoc")
            return
        if cmd == "cat" and not redirs:
            text, miss, dflt = self.render_heredoc(body, quoted)
            self.last_pipe = {"text": text, "miss": miss | dflt, "line": self.lineno}
            return
        # python3 - << EOF, bash << EOF ...: an interpreter script, not an installed file

    def render_heredoc(self, body, quoted):
        if quoted:
            return body, set(), set()
        miss, dflt = set(), set()
        text = expand_text(body, self.scope, True, miss, dflt)
        return text, miss, dflt

    def cmd_gb_managed_write(self, argv, redirs, text):
        if not argv or self.removal_ctx():
            return
        path, why = self.path_of(argv[0])
        mode = "".join(x for _, x in argv[1]) if len(argv) > 1 else "644"
        if not path:
            self.unres("managed-write", self.raw(argv[0]), "path %s" % why, "runtime-path")
            return
        frag = []
        if self.last_pipe and self.lineno - self.last_pipe["line"] < 60:
            frag = [l for l in self.last_pipe["text"].split("\n") if l.strip() and not l.startswith("#") and "$" not in l]
        e = {"kind": "file", "path": path, "mode": mode, "content": None, "opaque": True, "managed": True,
             "why_opaque": "generated by a function and stamped with a content hash (gb_managed_write)"}
        if frag:
            e["static_fragments"] = frag
        self.add(e)


def generate(installer, repo):
    with open(installer, "rb") as fh:
        data = fh.read()
    raw_lines = data.decode("utf-8", "replace").split("\n")
    g = Gen(repo)
    lines = [(l, n + 1) for n, l in enumerate(raw_lines)]
    g.scope = Scope(g.vars_scan(raw_lines))
    g.pre_arrays(lines)
    g.run(lines)
    entries = g.entries
    for e in entries:
        e.pop("func", None)
    by_path = {e["path"]: e for e in entries if e["kind"] == "file" and e.get("path")}
    # fold static sed edits into the file entry the installer itself wrote
    keep = []
    for e in entries:
        if e["kind"] == "line_edit" and e["path"] in by_path and not by_path[e["path"]].get("opaque") \
                and ((all(not o.get("unless_re") for o in e["ops"]) and all(o["op"] == "sed" for o in e["ops"]))
                     or all(o["op"] == "append_lines" and o.get("unless_re") for o in e["ops"])):
            # (idempotent guarded appends, e.g. the feed/connect network lines, are part of the file's final bytes)
            by_path[e["path"]].setdefault("transforms", []).extend(e["ops"])
            by_path[e["path"]]["steps"].extend(e["steps"])
            continue
        if e["kind"] == "line_edit":
            only_del = all(o["op"] == "sed" and all(st[0] == "d" for st in parse_sed(o["script"])) for o in e["ops"])
            e["revert"] = "none" if only_del else "restore_original"
        keep.append(e)
    entries = keep
    # unit_enable: wanted-by comes from the unit text, else multi-user.target
    for e in entries:
        if e["kind"] == "unit_enable":
            unit = by_path.get("/etc/systemd/system/" + e["name"])
            text = None
            if unit and unit.get("content"):
                text = unit["content"]
            elif unit and unit.get("source") and os.path.isfile(os.path.join(repo, unit["source"])):
                text = rd(os.path.join(repo, unit["source"]))
            m = re.search(r"^WantedBy=(\S+)", text or "", re.M)
            e["wanted_by"] = m.group(1) if m else "multi-user.target"
    for e in entries:
        if e["kind"] == "file" and e.get("source") and e.get("source_kind") != "build-output":
            with open(os.path.join(repo, e["source"]), "rb") as fh:
                e["source_sha256"] = sha(fh.read())
        if e["kind"] == "file" and e.get("content") is not None and PH_MODULE not in e["content"] and not e.get("transforms"):
            e["sha256"] = sha(e["content"].encode())
    cats = {}
    for u in g.unresolved:
        cats[u["category"]] = cats.get(u["category"], 0) + 1
    return {"version": VERSION,
            "generated_from": {"file": os.path.relpath(installer, repo) if installer.startswith(repo) else installer,
                               "sha256": sha(data), "lines": len(raw_lines)},
            "entries": entries, "unresolved": g.unresolved, "unresolved_by_category": cats,
            "removals": g.removals, "module_unloads": sorted(g.unloads),
            "remote_blocks": g.remote, "scratch_writes": len(g.scratch), "runtime_target_echo_writes": g.var_echo_writes,
            "multi_writer_files": [{"path": e["path"], "steps": e["steps"]} for e in entries if e.get("conflicting_writers")],
            "permission_changes": g.perms,
            "side_effect_commands": {k: len(v) for k, v in sorted(g.commands.items())}}


# ------------------------------------------------------------------ engine
def rp(root, p):
    return os.path.join(root, p.lstrip("/"))


def desired_bytes(e, repo, params=None):
    if e.get("opaque"):
        return None
    params = params or {}
    if e.get("source"):
        with open(os.path.join(repo, e["source"]), "rb") as fh:
            data = fh.read()
    else:
        data = (e.get("content") or "").encode()
    if e.get("transforms"):
        data = apply_ops(data.decode("utf-8", "replace"), e["transforms"]).encode()
    if PH_MODULE.encode() in data:
        data = data.replace(PH_MODULE.encode(), params.get("MODULE_DIR", repo).encode())
    return data


def load_state(root):
    p = rp(root, STATE_REL)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {"version": VERSION, "applied": {}, "created_dirs": [], "created_files": []}


def group_line(root, name):
    gid, used = 900, set()
    p = rp(root, "etc/group")
    if os.path.exists(p):
        for l in rd(p).splitlines():
            f = l.split(":")
            if len(f) > 2 and f[2].isdigit():
                used.add(int(f[2]))
    while gid in used:
        gid += 1
    return "%s:x:%d:" % (name, gid)


def le_text(e, text):
    return apply_ops(text, e["ops"])


def entry_state(e, root, repo, state, params=None):
    """returns (status, detail). status: new unchanged changed hand-edited opaque"""
    k = e["kind"]
    if k == "file":
        want = desired_bytes(e, repo, params)
        p = rp(root, e["path"])
        if want is None:
            fr = e.get("static_fragments")
            if fr and os.path.exists(p):
                have = set(rd(p).split("\n"))
                miss = [f for f in fr if f not in have]
                return "opaque", ("static lines all present" if not miss else "%d static line(s) missing, e.g. %r" % (len(miss), miss[0]))
            return "opaque", e.get("why_opaque", "")
        if not os.path.lexists(p):
            return "new", ""
        if e.get("create_only"):
            return "unchanged", "exists (create-only)"
        have = rd(p, "rb")
        if have == want and (os.stat(p).st_mode & 0o7777) == int(e["mode"], 8):
            return "unchanged", ""
        rec = state["applied"].get(e["id"])
        if rec and rec.get("sha256") != sha(have):
            return "hand-edited", "disk differs from what apply wrote"
        return "changed", ""
    if k == "line_edit":
        if e.get("incomplete"):
            return "opaque", e["incomplete"]
        p = rp(root, e["path"])
        if not os.path.exists(p):
            return ("unchanged", "absent: edit skipped") if e.get("only_if_exists") else ("new", "")
        have = rd(p)
        try:
            new = le_text(e, have)
        except Unsupported as ex:
            return "opaque", "unsupported: %s" % ex
        if new == have:
            return "unchanged", ""
        rec = state["applied"].get(e["id"])
        if rec and rec.get("sha256") != sha(have.encode()):
            return "hand-edited", "disk differs from what apply wrote"
        return "changed", ""
    if k == "dir":
        return ("unchanged" if os.path.isdir(rp(root, e["path"])) else "new"), ""
    if k == "group":
        p = rp(root, "etc/group")
        ok = any(l.split(":")[0] == e["name"] for l in (rd(p).splitlines() if os.path.exists(p) else []))
        return ("unchanged" if ok else "new"), ""
    if k == "user":
        p = rp(root, "etc/passwd")
        ok = any(l.split(":")[0] == e["name"] for l in (rd(p).splitlines() if os.path.exists(p) else []))
        return ("unchanged" if ok else "new"), ""
    if k == "unit_enable":
        p = rp(root, "etc/systemd/system/%s.wants/%s" % (e["wanted_by"], e["name"]))
        return ("unchanged" if os.path.islink(p) else "new"), ""
    if k == "kernel_module":
        p = rp(root, MODULES_REL)
        ok = e["name"] in (rd(p).split() if os.path.exists(p) else [])
        return ("unchanged" if ok else "new"), ""
    return "opaque", "unknown kind"


def label(e):
    return e.get("path") or e.get("name")


def cmd_plan(m, root, repo, out, params=None):
    st = load_state(root)
    counts = {}
    for e in m["entries"]:
        s, d = entry_state(e, root, repo, st, params)
        counts[s] = counts.get(s, 0) + 1
        out.write("%-11s %-12s %s %s\n" % (s, e["kind"], label(e), ("(" + d + ")") if d else ""))
        if e["kind"] == "file" and s in ("new", "changed", "hand-edited"):
            want = (desired_bytes(e, repo, params) or b"").decode("utf-8", "replace").splitlines(True)
            p = rp(root, e["path"])
            have = rd(p, "rb").decode("utf-8", "replace").splitlines(True) if os.path.exists(p) else []
            for l in difflib.unified_diff(have, want, "a" + e["path"], "b" + e["path"], n=1):
                out.write("    " + l.rstrip("\n") + "\n")
        if e["kind"] == "line_edit" and s in ("new", "changed", "hand-edited"):
            p = rp(root, e["path"])
            have = rd(p) if os.path.exists(p) else ""
            for l in difflib.unified_diff(have.splitlines(True), le_text(e, have).splitlines(True), "a" + e["path"], "b" + e["path"], n=0):
                out.write("    " + l.rstrip("\n") + "\n")
    out.write("summary: %s; unresolved edits (not in this plan): %d\n" % (json.dumps(counts, sort_keys=True), len(m.get("unresolved", []))))
    return counts


def refuse_real_root(root):
    if os.path.realpath(root) == "/":
        raise SystemExit("refused: this prototype never writes to /. Use --root DIR.")


def mkdirs(root, rel_dir, state):
    parts, cur = rel_dir.strip("/").split("/"), root
    for part in parts:
        cur = os.path.join(cur, part)
        if not os.path.isdir(cur):
            os.mkdir(cur)
            state["created_dirs"].append(os.path.relpath(cur, root))


def note_file(root, rel, state):
    if not os.path.lexists(rp(root, rel)) and rel not in state["created_files"]:
        state["created_files"].append(rel)


def write_state(root, state):
    mkdirs(root, os.path.dirname(STATE_REL), state)
    with open(rp(root, STATE_REL), "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)


def atomic_write(p, data, mode):
    tmp = p + ".gbnew"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.chmod(tmp, mode)
    os.replace(tmp, p)


def backup_of(root, state, e, p, rec):
    if e["id"] in state["applied"]:
        rec["backup"] = state["applied"][e["id"]].get("backup")
        rec["existed"] = state["applied"][e["id"]].get("existed", True)
        return
    rec["existed"] = os.path.lexists(p)
    if rec["existed"]:
        bp = os.path.join(BACKUP_REL, sha(e["id"].encode()))
        mkdirs(root, BACKUP_REL, state)
        with open(p, "rb") as a, open(rp(root, bp), "wb") as b:
            b.write(a.read())
        rec["backup"] = bp
        rec["orig_mode"] = os.stat(p).st_mode & 0o7777
        os.chmod(rp(root, bp), rec["orig_mode"])


def cmd_apply(m, root, repo, force, out, params=None):
    refuse_real_root(root)
    state = load_state(root)
    for e in m["entries"]:
        s, d = entry_state(e, root, repo, state, params)
        if s == "opaque":
            out.write("skip (opaque) %s\n" % label(e))
            continue
        if s == "unchanged":
            continue
        if s == "hand-edited" and not force:
            out.write("REFUSED hand-edited %s (use --force)\n" % e["path"])
            continue
        k, rec = e["kind"], {"kind": e["kind"]}
        if k == "file":
            p = rp(root, e["path"])
            mkdirs(root, os.path.dirname(e["path"]), state)
            backup_of(root, state, e, p, rec)
            data = desired_bytes(e, repo, params)
            atomic_write(p, data, int(e["mode"], 8))
            rec["sha256"] = sha(data)
            rec["path"] = e["path"]
        elif k == "line_edit":
            p = rp(root, e["path"])
            mkdirs(root, os.path.dirname(e["path"]), state)
            backup_of(root, state, e, p, rec)
            have = rd(p) if os.path.exists(p) else ""
            if not rec["existed"]:
                note_file(root, e["path"].lstrip("/"), state)
            new = le_text(e, have).encode()
            atomic_write(p, new, rec.get("orig_mode", 0o644))
            rec["sha256"], rec["path"], rec["revert"] = sha(new), e["path"], e.get("revert", "restore_original")
        elif k == "dir":
            mkdirs(root, e["path"], state)
        elif k in ("group", "user"):
            rel = "etc/group" if k == "group" else "etc/passwd"
            line = group_line(root, e["name"]) if k == "group" else "%s:x:%d:900::/nonexistent:/usr/sbin/nologin" % (e["name"], 990)
            note_file(root, rel, state)
            mkdirs(root, "etc", state)
            with open(rp(root, rel), "a") as fh:
                fh.write(line + "\n")
            rec["file"], rec["line"] = rel, line
        elif k == "unit_enable":
            d_ = "etc/systemd/system/%s.wants" % e["wanted_by"]
            mkdirs(root, d_, state)
            os.symlink("/etc/systemd/system/" + e["name"], rp(root, d_ + "/" + e["name"]))
            rec["path"] = d_ + "/" + e["name"]
        elif k == "kernel_module":
            note_file(root, MODULES_REL, state)
            mkdirs(root, "run", state)
            with open(rp(root, MODULES_REL), "a") as fh:
                fh.write(e["name"] + "\n")
            rec["file"], rec["line"] = MODULES_REL, e["name"]
        state["applied"][e["id"]] = rec
        state.setdefault("order", []).append(e["id"])
        write_state(root, state)
        out.write("applied %s %s\n" % (k, label(e)))
    write_state(root, state)


def cmd_verify(m, root, repo, out, params=None, strict_opaque=False):
    st, bad = load_state(root), 0
    for e in m["entries"]:
        s, d = entry_state(e, root, repo, st, params)
        if s in ("new", "changed", "hand-edited"):
            bad += 1
            out.write("%s %s\n" % ({"new": "MISSING"}.get(s, s.upper()), label(e)))
        elif s == "opaque" and e["kind"] == "file":
            p = rp(root, e["path"])
            if strict_opaque and not os.path.exists(p):
                bad += 1
                out.write("MISSING (opaque) %s\n" % e["path"])
            elif "missing" in d and "static" in d:
                bad += 1
                out.write("DRIFT (opaque) %s: %s\n" % (e["path"], d))
    out.write("verify: %s\n" % ("ok" if not bad else "%d problem(s)" % bad))
    return bad


def drop_line(path, line):
    if not os.path.exists(path):
        return
    ls = rd(path).split("\n")
    if line in ls:
        ls.remove(line)
    with open(path, "w") as fh:
        fh.write("\n".join(ls))


def cmd_uninstall(m, root, force, out):
    refuse_real_root(root)
    state = load_state(root)
    ids = [e["id"] for e in m["entries"]]
    order = [i for i in reversed(state.get("order", [])) if i in ids]  # only what the manifest created
    kept = []
    for i in order:
        rec = state["applied"][i]
        if rec["kind"] in ("file", "line_edit"):
            p = rp(root, rec["path"])
            bk = rp(root, rec["backup"]) if rec.get("backup") else None
            if os.path.lexists(p):
                if sha(rd(p, "rb")) != rec["sha256"] and not force:
                    out.write("KEPT hand-edited %s\n" % rec["path"])
                    kept.append(i)
                    continue
                if rec.get("revert") == "none":
                    if bk and os.path.exists(bk):
                        os.unlink(bk)
                elif bk and os.path.exists(bk):
                    os.replace(bk, p)
                elif rec.get("existed", False) is False:
                    os.unlink(p)
                else:
                    os.unlink(p)
            elif bk and os.path.exists(bk):
                os.unlink(bk)
        elif rec["kind"] in ("group", "user", "kernel_module"):
            drop_line(rp(root, rec["file"]), rec["line"])
        elif rec["kind"] == "unit_enable":
            if os.path.islink(rp(root, rec["path"])):
                os.unlink(rp(root, rec["path"]))
        out.write("removed %s\n" % i)
        del state["applied"][i]
    state["order"] = [i for i in state.get("order", []) if i in state["applied"]]
    if kept:
        write_state(root, state)
        return
    for rel in state["created_files"]:  # files we created that are now empty
        p = rp(root, rel)
        if os.path.isfile(p) and not rd(p).strip():
            os.unlink(p)
    sp = rp(root, STATE_REL)
    if os.path.exists(sp):
        os.unlink(sp)
    for rel in sorted(state["created_dirs"], key=lambda r: -r.count("/")):
        try:
            os.rmdir(rp(root, rel))
        except OSError:
            out.write("left non-empty dir %s\n" % rel)


# --------------------------------------------------------------- coverage
def covered(e, m):
    """Is there an uninstall step (rm / sed in a removal function, systemctl disable, rmmod)?"""
    rem = [r for r in m.get("removals", []) if r.get("removal_ctx")]
    k = e["kind"]
    if k == "kernel_module":
        return e["name"] in m.get("module_unloads", [])
    if k in ("group", "user"):
        return any(r["target"] == "group:" + e["name"] for r in rem) if k == "group" else False
    if k == "dir":
        return None
    path = e.get("path")
    if k == "unit_enable":
        if any(r["target"] == "unit:" + e["name"] for r in rem):
            return True
        path = "/etc/systemd/system/" + e["name"]
    if k == "line_edit" and e.get("revert") == "none":
        return None
    for r in rem:
        t = r["target"]
        if t.startswith("unit:"):
            continue
        if r.get("sed"):
            if k == "line_edit" and t == path:
                return True
            continue
        if fnmatch.fnmatch(path, t):
            return True
        if r.get("recursive") and path.startswith(t.rstrip("/") + "/"):
            return True
    return False


def cmd_coverage(m, out, notes=None):
    """notes: [{"match": glob over the entry label, "why": reviewed reason}] for removals the static
    reader cannot see (a path list read at run time) or intentional leftovers."""
    unc, cov, na, rev = [], 0, 0, 0
    for e in m["entries"]:
        c = covered(e, m)
        if c is None:
            na += 1
        elif c:
            cov += 1
        else:
            note = next((n for n in (notes or []) if fnmatch.fnmatch(label(e), n["match"])), None)
            if note:
                rev += 1
                out.write("reviewed  %-12s %s  (%s)\n" % (e["kind"], label(e), note["why"]))
            else:
                unc.append(e)
    for e in unc:
        out.write("UNCOVERED %-12s %s  (installed at %s)\n" % (e["kind"], label(e), ", ".join(e["steps"][:2])))
    out.write("coverage: %d covered, %d not covered, %d reviewed, %d not applicable\n" % (cov, len(unc), rev, na))
    return unc


def cmd_summary(m, out):
    out.write("%d entries, %d unresolved, %d remote blocks, %d scratch writes\n" % (len(m["entries"]), len(m["unresolved"]), len(m.get("remote_blocks", [])), m.get("scratch_writes", 0)))
    kinds = {}
    for e in m["entries"]:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    out.write("entries by kind: %s\n" % json.dumps(kinds, sort_keys=True))
    out.write("opaque files: %d\n" % sum(1 for e in m["entries"] if e["kind"] == "file" and e.get("opaque")))
    out.write("unresolved by category: %s\n" % json.dumps(m.get("unresolved_by_category", {}), sort_keys=True))
    for u in m["unresolved"]:
        out.write("  [%s] %s %s (%s) %s\n" % (u["category"], u["what"], u["target"], u["step"], u["why"]))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="gbmanifest")
    ap.add_argument("cmd", choices=["generate", "plan", "apply", "verify", "uninstall", "coverage", "summary"])
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    ap.add_argument("--installer")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--root")
    ap.add_argument("--module-dir", default=None, help="value of the installer's $MODULE_DIR (default: --repo)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--strict-opaque", action="store_true")
    ap.add_argument("--notes", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "coverage_notes.json"))
    a = ap.parse_args(argv)
    mf = a.manifest or os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")
    params = {"MODULE_DIR": a.module_dir or a.repo}
    if a.cmd == "generate":
        m = generate(a.installer or os.path.join(a.repo, "greenboost_setup.sh"), a.repo)
        with open(mf, "w") as fh:
            json.dump(m, fh, indent=1)
        print("%d entries, %d unresolved edits -> %s" % (len(m["entries"]), len(m["unresolved"]), mf))
        return 0
    m = json.load(open(mf))
    if a.cmd == "coverage":
        notes = json.load(open(a.notes)) if os.path.exists(a.notes) else None
        return 1 if cmd_coverage(m, sys.stdout, notes) else 0
    if a.cmd == "summary":
        cmd_summary(m, sys.stdout)
        return 0
    if not a.root:
        ap.error("--root is required")
    if a.cmd == "plan":
        cmd_plan(m, a.root, a.repo, sys.stdout, params)
    elif a.cmd == "apply":
        cmd_apply(m, a.root, a.repo, a.force, sys.stdout, params)
    elif a.cmd == "verify":
        return 1 if cmd_verify(m, a.root, a.repo, sys.stdout, params, a.strict_opaque) else 0
    else:
        cmd_uninstall(m, a.root, a.force, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
