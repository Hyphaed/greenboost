"""Tests for tools/manifest/gbmanifest.py. Fake root only: no sudo, no systemctl, no modprobe,
nothing outside a temp directory under ~/.cache (disk, not RAM-backed /tmp)."""
import io, json, os, sys, tempfile, unittest
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import gbmanifest as g

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

FIXTURE = r'''#!/bin/bash
MODULE_DIR="/nonexistent"
GB_CONF="/etc/greenboost/main.conf"
install_things() {
    groupadd --system greenboost 2>/dev/null
    install -m 755 "$MODULE_DIR/tool.sh" /usr/local/sbin/gb-tool
    cat > /etc/udev/rules.d/99-x.rules << 'UDEV'
ACTION=="add", SUBSYSTEM=="pci"
UDEV
    cat > "$GB_CONF" << 'CF'
key=1
CF
    cat > /etc/profile.d/t.sh << PROF
export X=$HOME
PROF
    ssh host "sudo -n bash -s" << 'REMOTE'
install -m 755 netd /usr/local/bin/remote-only
REMOTE
    sed -i 's/a/b/' /etc/default/grub
    systemctl enable --now greenboost-a.service
    modprobe greenboost
}
'''

# A second fixture with the constructs round 2 models: function-scoped variables, arrays and loops,
# cp+mv, folded and standalone sed edits, appends with a guard, group writes, tmp+mv, conflicts.
FIX2 = r'''#!/bin/bash
set -euo pipefail
MODULE_DIR="/nonexistent"
SHIM_DEST="/usr/local/lib"
GB_CONF_DIR="/etc/gbx"
_files=(
    a.py b.py
    c.py )
purge_all() {
    rm -f /etc/gbx/removed.conf
    for f in /etc/systemd/system/gbx*.service /etc/gbx/old.conf; do
        [[ -f "$f" ]] && rm -f "$f"
    done
    rm -rf /usr/local/lib/gbx
}
install_all() {
    local dest="$GB_CONF_DIR/main.conf"
    cat > "$dest" << 'CF'
key=1
CF
    cat > /etc/gbx/rendered.conf << RC
lib=$SHIM_DEST/libx.so
escaped=\$HOME
RC
    cat > /etc/gbx/runtime.conf << RT
cpus=${GB_CPUS}
static=yes
RT
    cat > /etc/gbx/removed.conf << 'X'
r=1
X
    cat > /etc/gbx/kept.conf << 'X'
k=1
X
    mkdir -p /usr/local/lib/gbx
    for _f in "${_files[@]}"; do
        install -m 644 "$MODULE_DIR/$_f" "/usr/local/lib/gbx/$_f"
    done
    for _lib in liba.so libb.so; do
        cp "$MODULE_DIR/$_lib" "$SHIM_DEST/.$_lib.tmp.$$"
        mv -f "$SHIM_DEST/.$_lib.tmp.$$" "$SHIM_DEST/$_lib"
    done
    install -m 644 "$MODULE_DIR/unit.service" /etc/systemd/system/gbx.service
    sed -i 's/^TimeoutStartSec=60$/TimeoutStartSec=300/' /etc/systemd/system/gbx.service
    systemctl enable gbx.service
    sed -i '/OLD_KEY/d' /etc/gbx/shared.conf
    sed -i '/Environment="X=1"/d' /etc/gbx/svc.service
    sed -i '/^\[Service\]/a Environment="X=1"' /etc/gbx/svc.service
    sed -i 's/q8_0/q4_0/' /etc/gbx/svc.service
    if ! grep -q 'gbx-marker' /etc/gbx/local.rules 2>/dev/null; then
        {
            echo "# gbx-marker"
            echo "allow /usr/lib/gbx r,"
        } >> /etc/gbx/local.rules
    fi
    {
        cat << DROPIN
[Service]
Environment="A=1"
Environment="B=${GB_B}"
DROPIN
        getent group gbx >/dev/null 2>&1 && echo "SupplementaryGroups=gbx"
    } | gb_managed_write /etc/gbx/dropin.conf 644
    local _t; _t="$(mktemp /etc/sudoers.d/.gbx.XXXXXX)"
    printf '%s ALL=(root) NOPASSWD: /usr/local/sbin/gbx\n' "$_user" > "$_t"
    chmod 440 "$_t"
    if visudo -c -f "$_t" && mv -f "$_t" /etc/sudoers.d/gbx; then :; fi
    ssh -o BatchMode=yes host \
        "sudo -n bash -s" << 'REMOTE'
install -m 755 netd /usr/local/bin/remote-only
REMOTE
    echo "autoload" > /etc/modules-load.d/gbx.conf
    sed -i "s/^x=.*/x=${RUNTIME}/" /etc/gbx/rt.conf
}
conf_a() {
    cat > /etc/gbx/dup.conf << 'X'
one
X
}
conf_b() {
    cat > /etc/gbx/dup.conf << 'X'
two
X
}
'''


def snap(root):
    out = {}
    for d, ds, fs in os.walk(root):
        for n in ds + fs:
            p = os.path.join(d, n)
            rel = os.path.relpath(p, root)
            if os.path.islink(p):
                out[rel] = ("link", os.readlink(p))
            elif os.path.isdir(p):
                out[rel] = ("dir",)
            else:
                out[rel] = ("file", g.rd(p, "rb"), os.stat(p).st_mode & 0o7777)
    return out


class Base(unittest.TestCase):
    def setUp(self):
        wd = os.path.expanduser("~/.cache/gbm-tests")  # disk, not RAM-backed /tmp
        os.makedirs(wd, exist_ok=True)
        self.td = tempfile.TemporaryDirectory(dir=wd)
        self.addCleanup(self.td.cleanup)
        base = self.td.name
        self.repo = os.path.join(base, "repo")
        self.root = os.path.join(base, "root")
        os.mkdir(self.repo)
        os.mkdir(self.root)

    def put_repo(self, name, text):
        with open(os.path.join(self.repo, name), "w") as f:
            f.write(text)

    def put_root(self, rel, text, mode=None):
        p = os.path.join(self.root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
        if mode:
            os.chmod(p, mode)

    def read_root(self, rel):
        return g.rd(os.path.join(self.root, rel))

    def run_(self, fn, *a):
        out = io.StringIO()
        r = fn(*a, out)
        return r, out.getvalue()

    def apply(self, m, force=False):
        return self.run_(lambda o: g.cmd_apply(m, self.root, self.repo, force, o))[1]

    def plan(self, m):
        return self.run_(lambda o: g.cmd_plan(m, self.root, self.repo, o))

    def uninstall(self, m, force=False):
        return self.run_(lambda o: g.cmd_uninstall(m, self.root, force, o))[1]


class T(Base):
    def setUp(self):
        super().setUp()
        self.put_repo("tool.sh", "#!/bin/sh\necho hi\n")
        self.inst = os.path.join(self.repo, "setup.sh")
        with open(self.inst, "w") as f:
            f.write(FIXTURE)
        self.m = g.generate(self.inst, self.repo)

    def test_generate_static(self):
        ids = {e["id"] for e in self.m["entries"]}
        for want in ("group:greenboost", "file:/usr/local/sbin/gb-tool", "file:/etc/udev/rules.d/99-x.rules",
                     "file:/etc/greenboost/main.conf", "unit_enable:greenboost-a.service", "kernel_module:greenboost"):
            self.assertIn(want, ids)
        self.assertNotIn("file:/usr/local/bin/remote-only", ids)  # inside a remote heredoc
        self.assertEqual(len(self.m["remote_blocks"]), 1)
        byid = {e["id"]: e for e in self.m["entries"]}
        self.assertTrue(byid["file:/etc/profile.d/t.sh"]["opaque"])  # unquoted heredoc with a runtime $HOME
        self.assertEqual(byid["file:/usr/local/sbin/gb-tool"]["source"], "tool.sh")
        # round 1 left this sed as "unresolved"; round 2 models it as a line edit on a file the installer does not own
        self.assertEqual(self.m["unresolved"], [])
        le = byid["line_edit:/etc/default/grub"]
        self.assertEqual(le["ops"][0]["script"], "s/a/b/")

    def test_gate_zero_tree_diff(self):
        self.put_root("etc/group", "root:x:0:\n")
        self.put_root("usr/local/sbin/gb-tool", "old user file\n")  # pre-existing file must be restored
        self.put_root("etc/default/grub", "GRUB=a\n")     # pre-existing edited file must be restored
        before = snap(self.root)
        self.apply(self.m)
        self.assertNotEqual(before, snap(self.root))
        self.assertEqual(self.read_root("usr/local/sbin/gb-tool"), "#!/bin/sh\necho hi\n")
        self.assertEqual(self.read_root("etc/default/grub"), "GRUB=b\n")
        _, v = self.run_(lambda o: g.cmd_verify(self.m, self.root, self.repo, o))
        self.assertIn("verify: ok", v)
        self.uninstall(self.m)
        self.assertEqual(before, snap(self.root))

    def test_gate_empty_root_zero_diff(self):
        self.apply(self.m)
        self.uninstall(self.m)
        self.assertEqual({}, snap(self.root))

    def test_second_plan_empty(self):
        self.put_root("etc/default/grub", "ab\n")
        self.apply(self.m)
        c, _ = self.plan(self.m)
        self.assertEqual(set(c) - {"unchanged", "opaque"}, set())

    def test_hand_edit_detected_and_kept(self):
        self.apply(self.m)
        p = os.path.join(self.root, "etc/greenboost/main.conf")
        with open(p, "w") as f:
            f.write("key=2\n")
        c, txt = self.plan(self.m)
        self.assertEqual(c.get("hand-edited"), 1)
        a = self.apply(self.m)
        self.assertIn("REFUSED", a)
        self.assertEqual(g.rd(p), "key=2\n")
        self.assertIn("KEPT", self.uninstall(self.m))
        self.assertEqual(g.rd(p), "key=2\n")

    def test_plan_touches_nothing(self):
        self.plan(self.m)
        self.assertEqual({}, snap(self.root))

    def test_uninstall_only_what_manifest_created(self):
        self.apply(self.m)
        stray = os.path.join(self.root, "etc/greenboost/user-note")
        with open(stray, "w") as f:
            f.write("mine\n")
        self.uninstall(self.m)
        self.assertTrue(os.path.exists(stray))

    def test_refuses_real_root(self):
        with self.assertRaises(SystemExit):
            g.cmd_apply(self.m, "/", self.repo, False, io.StringIO())
        with self.assertRaises(SystemExit):
            g.cmd_uninstall(self.m, "/", False, io.StringIO())


class Round2(Base):
    def setUp(self):
        super().setUp()
        for f in ("a.py", "b.py", "c.py", "liba.so", "libb.so"):
            self.put_repo(f, "# %s\n" % f)
        self.put_repo("unit.service", "[Service]\nTimeoutStartSec=60\n[Install]\nWantedBy=sysinit.target\n")
        self.inst = os.path.join(self.repo, "setup.sh")
        with open(self.inst, "w") as f:
            f.write(FIX2)
        self.m = g.generate(self.inst, self.repo)
        self.by = {e["id"]: e for e in self.m["entries"]}

    # ---- generation: what becomes an entry
    def test_scoped_variables_and_heredocs(self):
        e = self.by["file:/etc/gbx/main.conf"]            # "$GB_CONF_DIR/main.conf" held in a function-local
        self.assertEqual(e["content"], "key=1\n")
        r = self.by["file:/etc/gbx/rendered.conf"]        # unquoted heredoc, constants expand, \$ stays literal
        self.assertEqual(r["content"], "lib=/usr/local/lib/libx.so\nescaped=$HOME\n")
        rt = self.by["file:/etc/gbx/runtime.conf"]         # a runtime value: opaque, static lines still checkable
        self.assertTrue(rt["opaque"])
        self.assertEqual(rt["runtime_vars"], ["GB_CPUS"])
        self.assertIn("static=yes", rt["static_fragments"])

    def test_array_and_loop_expansion(self):
        for f in ("a.py", "b.py", "c.py"):
            e = self.by["file:/usr/local/lib/gbx/" + f]
            self.assertEqual(e["source"], f)
            self.assertEqual(e["mode"], "644")
        for lib in ("liba.so", "libb.so"):                # cp to a temp name, then mv: one install
            e = self.by["file:/usr/local/lib/" + lib]
            self.assertEqual(e["source"], lib)
            self.assertEqual(e["source_kind"], "build-output")
        self.assertNotIn("file:/usr/local/bin/remote-only", self.by)

    def test_folded_sed_and_unit_wanted_by(self):
        e = self.by["file:/etc/systemd/system/gbx.service"]
        self.assertEqual(e["transforms"], [{"op": "sed", "script": "s/^TimeoutStartSec=60$/TimeoutStartSec=300/"}])
        self.assertNotIn("line_edit:/etc/systemd/system/gbx.service", self.by)
        self.assertEqual(self.by["unit_enable:gbx.service"]["wanted_by"], "sysinit.target")
        self.assertIn(b"TimeoutStartSec=300", g.desired_bytes(e, self.repo))

    def test_line_edits(self):
        shared = self.by["line_edit:/etc/gbx/shared.conf"]
        self.assertEqual(shared["revert"], "none")        # pure deletion: a scrub, never re-added on uninstall
        svc = self.by["line_edit:/etc/gbx/svc.service"]   # two seds merged, in order
        self.assertEqual([o["script"] for o in svc["ops"]], ['/Environment="X=1"/d', '/^\\[Service\\]/a Environment="X=1"', "s/q8_0/q4_0/"])
        self.assertEqual(svc["revert"], "restore_original")
        rules = self.by["line_edit:/etc/gbx/local.rules"]  # `{ echo; echo; } >>` behind an `if ! grep -q`
        self.assertEqual(rules["ops"][0]["lines"], ["# gbx-marker", "allow /usr/lib/gbx r,"])
        self.assertEqual(rules["ops"][0]["unless_re"], "gbx-marker")
        self.assertFalse(rules["only_if_exists"])

    def test_runtime_values_stay_unresolved_with_a_reason(self):
        un = [u for u in self.m["unresolved"] if u["target"] == "/etc/gbx/rt.conf"]
        self.assertEqual(len(un), 1)
        self.assertEqual(un[0]["category"], "runtime-value")
        self.assertIn("RUNTIME", un[0]["why"])
        self.assertNotIn("line_edit:/etc/gbx/rt.conf", self.by)
        self.assertEqual(self.m["unresolved_by_category"], {"runtime-value": 1})

    def test_group_write_and_tmp_mv(self):
        d = self.by["file:/etc/gbx/dropin.conf"]
        self.assertTrue(d["opaque"] and d["managed"])
        self.assertIn('Environment="A=1"', d["static_fragments"])
        self.assertNotIn('Environment="B=${GB_B}"', d["static_fragments"])
        s = self.by["file:/etc/sudoers.d/gbx"]            # printf > "$tmp"; chmod 440 "$tmp"; mv -f "$tmp" dest
        self.assertEqual(s["mode"], "440")
        self.assertTrue(s["opaque"])
        self.assertEqual(self.by["file:/etc/modules-load.d/gbx.conf"]["content"], "autoload\n")

    def test_two_writers_with_different_content_become_opaque(self):
        e = self.by["file:/etc/gbx/dup.conf"]
        self.assertTrue(e["opaque"])
        self.assertEqual(len(e["conflicting_writers"]), 1)
        self.assertEqual([x["path"] for x in self.m["multi_writer_files"]], ["/etc/gbx/dup.conf"])

    def test_remote_heredoc_over_a_continued_ssh_line(self):
        self.assertEqual(len(self.m["remote_blocks"]), 1)

    # ---- engine: line edits on a fake root
    def populate(self):
        self.put_root("etc/gbx/shared.conf", "keep=1\nOLD_KEY=stale\nkeep2=2\n")
        self.put_root("etc/gbx/svc.service", "[Unit]\n[Service]\nEnvironment=KV=q8_0\nExecStart=/x\n", 0o600)
        self.put_root("etc/gbx/local.rules", "# mine\n/a/b r,")   # no trailing newline
        self.put_root("etc/group", "root:x:0:\n")

    def test_apply_edits_lines_and_second_plan_is_empty(self):
        self.populate()
        self.apply(self.m)
        self.assertEqual(self.read_root("etc/gbx/shared.conf"), "keep=1\nkeep2=2\n")
        self.assertEqual(self.read_root("etc/gbx/svc.service"),
                         '[Unit]\n[Service]\nEnvironment="X=1"\nEnvironment=KV=q4_0\nExecStart=/x\n')
        self.assertEqual(self.read_root("etc/gbx/local.rules"), "# mine\n/a/b r,\n# gbx-marker\nallow /usr/lib/gbx r,\n")
        self.assertEqual(os.stat(os.path.join(self.root, "etc/gbx/svc.service")).st_mode & 0o7777, 0o600)  # mode kept
        c, _ = self.plan(self.m)
        self.assertEqual(set(c) - {"unchanged", "opaque"}, set())
        again = self.apply(self.m)
        self.assertNotIn("applied line_edit", again)
        self.assertEqual(self.read_root("etc/gbx/local.rules").count("gbx-marker"), 1)  # the guard keeps appends from stacking

    def test_uninstall_restores_edited_files_byte_for_byte(self):
        self.populate()
        before = snap(self.root)
        self.apply(self.m)
        self.uninstall(self.m)
        after = snap(self.root)
        # the scrub (a pure delete) is not undone: re-adding a stale line would be wrong
        scrubbed = dict(before)
        scrubbed["etc/gbx/shared.conf"] = ("file", b"keep=1\nkeep2=2\n", before["etc/gbx/shared.conf"][2])
        self.assertEqual(scrubbed, after)

    def test_edit_of_an_absent_file_is_skipped_and_an_append_creates_then_removes_it(self):
        self.put_root("etc/group", "root:x:0:\n")
        c, txt = self.plan(self.m)
        self.assertIn("absent: edit skipped", txt)
        before = snap(self.root)
        self.apply(self.m)
        self.assertEqual(self.read_root("etc/gbx/local.rules"), "# gbx-marker\nallow /usr/lib/gbx r,\n")
        self.uninstall(self.m)
        self.assertEqual(before, snap(self.root))

    def test_hand_edited_line_edit_is_kept_and_not_overwritten(self):
        self.populate()
        self.apply(self.m)
        p = os.path.join(self.root, "etc/gbx/svc.service")
        with open(p, "a") as f:
            f.write("Environment=MINE=1\n")
        # the edit would still change something only if its result differed; touch the part an op rewrites
        with open(p) as f:
            t = f.read().replace("q4_0", "q8_0")
        with open(p, "w") as f:
            f.write(t)
        c, _ = self.plan(self.m)
        self.assertEqual(c.get("hand-edited"), 1)
        self.assertIn("REFUSED", self.apply(self.m))
        self.assertIn("KEPT", self.uninstall(self.m))
        self.assertIn("MINE=1", g.rd(p))

    def test_opaque_file_is_never_written_but_its_static_lines_are_checked(self):
        self.put_root("etc/group", "root:x:0:\n")
        self.apply(self.m)
        self.assertFalse(os.path.exists(os.path.join(self.root, "etc/gbx/runtime.conf")))
        self.put_root("etc/gbx/runtime.conf", "cpus=8\nstatic=yes\n")
        _, txt = self.plan(self.m)
        self.assertIn("static lines all present", txt)
        self.put_root("etc/gbx/runtime.conf", "cpus=8\n")
        _, txt = self.plan(self.m)
        self.assertIn("static line(s) missing", txt)
        bad, out = self.run_(lambda o: g.cmd_verify(self.m, self.root, self.repo, o))
        self.assertIn("DRIFT (opaque) /etc/gbx/runtime.conf", out)

    def test_group_user_modules_units_still_modelled_as_plain_files(self):
        m = g.generate(self.inst, self.repo)
        self.assertIn("unit_enable:gbx.service", {e["id"] for e in m["entries"]})
        self.put_root("etc/group", "root:x:0:\n")
        self.apply(m)
        link = os.path.join(self.root, "etc/systemd/system/sysinit.target.wants/gbx.service")
        self.assertTrue(os.path.islink(link))

    def test_module_dir_parameter_is_rendered(self):
        self.put_repo("w.sh", "x\n")
        src = os.path.join(self.repo, "w.sh")
        with open(src, "w") as f:
            f.write('#!/bin/bash\nMODULE_DIR="/x"\nf() {\n    cat > /usr/local/bin/gbw << WRAP\nGB_SETUP="$MODULE_DIR/setup.sh"\nWRAP\n}\n')
        m = g.generate(src, self.repo)
        e = m["entries"][0]
        self.assertFalse(e["opaque"])
        out = io.StringIO()
        g.cmd_apply(m, self.root, self.repo, False, out, {"MODULE_DIR": "/home/x/gb"})
        self.assertEqual(self.read_root("usr/local/bin/gbw"), 'GB_SETUP="/home/x/gb/setup.sh"\n')

    # ---- coverage (INS-13, mechanical)
    def test_coverage_lists_what_no_uninstall_step_touches(self):
        out = io.StringIO()
        unc = g.cmd_coverage(self.m, out)
        unc_paths = {g.label(e) for e in unc}
        self.assertIn("/etc/gbx/kept.conf", unc_paths)               # never removed
        self.assertIn("/etc/sudoers.d/gbx", unc_paths)
        self.assertNotIn("/etc/gbx/removed.conf", unc_paths)         # rm -f
        self.assertNotIn("/usr/local/lib/gbx/a.py", unc_paths)       # under `rm -rf /usr/local/lib/gbx`
        self.assertNotIn("/etc/systemd/system/gbx.service", unc_paths)  # glob in a for-loop in purge_all
        self.assertNotIn("unit_enable:gbx.service", {e["id"] for e in unc})
        notes = [{"match": "/etc/gbx/kept.conf", "why": "reviewed"}]
        out = io.StringIO()
        unc2 = g.cmd_coverage(self.m, out, notes)
        self.assertNotIn("/etc/gbx/kept.conf", {g.label(e) for e in unc2})
        self.assertIn("reviewed", out.getvalue())


class SedModel(unittest.TestCase):
    def test_statement_forms(self):
        t = "a\nOLD=1\n[Service]\nx=q8_0\n}\n"
        self.assertEqual(g.apply_sed(t, g.parse_sed("/OLD/d")), "a\n[Service]\nx=q8_0\n}\n")
        self.assertEqual(g.apply_sed(t, g.parse_sed(r"\|OLD|d")), "a\n[Service]\nx=q8_0\n}\n")
        self.assertEqual(g.apply_sed(t, g.parse_sed("s/q8_0/q4_0/")), "a\nOLD=1\n[Service]\nx=q4_0\n}\n")
        self.assertEqual(g.apply_sed(t, g.parse_sed(r"/^\[Service\]/a Environment=A")), "a\nOLD=1\n[Service]\nEnvironment=A\nx=q8_0\n}\n")
        self.assertEqual(g.apply_sed(t, g.parse_sed(r"/^}$/i\  #include <x>")), "a\nOLD=1\n[Service]\nx=q8_0\n  #include <x>\n}\n")
        # GNU one-liner: \n inside a/i text is a newline, and one script may mix statements
        two = g.apply_sed("[Service]\nOLD=1\n", g.parse_sed("/OLD/d\n/^\\[Service\\]/a A=1\\nB=2"))
        self.assertEqual(two, "[Service]\nA=1\nB=2\n")

    def test_posix_classes_and_anchors(self):
        t = "greenboost  \ngreenboost_x\n"
        self.assertEqual(g.apply_sed(t, g.parse_sed(r"/^greenboost[[:space:]]*$/d")), "greenboost_x\n")

    def test_outside_the_subset_is_refused_not_guessed(self):
        for bad in ("y/abc/xyz/", "1,3d", r"s/a/\1/", "s/a/b/w out", r"/\<w\>/d"):
            with self.assertRaises(g.Unsupported):
                g.parse_sed(bad)


class RealInstaller(Base):
    def setUp(self):
        super().setUp()
        self.repo = REPO
        self.inst = os.path.join(REPO, "greenboost_setup.sh")
        self.m = g.generate(self.inst, REPO)
        self.ids = {e["id"] for e in self.m["entries"]}

    def test_real_installer_generates_and_is_richer_than_round_1(self):
        # round 1: 26 entries, 7 opaque, 31 unresolved
        self.assertGreater(len(self.m["entries"]), 150)
        self.assertLess(len(self.m["unresolved"]), 40)
        kinds = {}
        for e in self.m["entries"]:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        self.assertGreaterEqual(kinds["line_edit"], 8)
        self.assertGreaterEqual(kinds["dir"], 10)
        for want in ("file:/etc/systemd/system/greenboost-boot-guard.service",
                     "file:/etc/systemd/system/ollama.service.d/99-greenboost.conf",
                     "file:/etc/sudoers.d/greenboost-diag-read", "file:/etc/tmpfiles.d/greenboost.conf",
                     "file:/etc/profile.d/greenboost.sh", "file:/usr/local/bin/greenboost",
                     "file:/usr/local/lib/greenboost/gb_supervisor.py", "group:greenboost"):
            self.assertIn(want, self.ids)
        # the round-1 manifest hard-coded multi-user.target for every unit; the boot guard wants sysinit.target
        bg = next(e for e in self.m["entries"] if e["id"] == "unit_enable:greenboost-boot-guard.service")
        self.assertEqual(bg["wanted_by"], "sysinit.target")
        # the TimeoutStartSec sed is folded into the file entry, so the desired bytes carry 300
        unit = next(e for e in self.m["entries"] if e["id"] == "file:/etc/systemd/system/greenboost-boot-guard.service")
        self.assertIn(b"TimeoutStartSec=300", g.desired_bytes(unit, REPO))

    def test_known_unresolved_stay_unresolved_with_a_category(self):
        cats = self.m["unresolved_by_category"]
        self.assertTrue(set(cats) <= {"runtime-value", "runtime-path", "runtime-user", "dynamic-set", "unsupported", "unsupported-sed"})
        targets = {u["target"] for u in self.m["unresolved"]}
        self.assertIn("/etc/systemd/system/ollama.service", targets)  # inserts values from detect_hardware
        self.assertIn("$_f", targets)                                  # other people's drop-ins: a glob at run time

    def test_one_writer_for_the_sysctl_file(self):
        # round 4: install, `tune` and `tune-sysctl` share _tune_persist_sysctl (the long block is gone)
        # round 5: the function has two writes (GB_VM_SYSCTL=0 opt-out line, and the 3 keys); still ONE function
        mw = {x["path"]: x for x in self.m["multi_writer_files"]}
        w = mw.get("/etc/sysctl.d/99-zzz-greenboost.conf")
        if w:
            self.assertEqual({s.split(":")[0] for s in w["steps"]}, {"_tune_persist_sysctl"})
        e = [x for x in self.m["entries"] if x.get("path") == "/etc/sysctl.d/99-zzz-greenboost.conf" and x["kind"] == "file"][0]
        self.assertNotIn("overcommit", e["content"])
        self.assertNotIn("max_map_count", e["content"])

    def test_no_nvidia_preserve_memory_file_is_written(self):
        # INS-17: dead config removed from install; uninstall still removes older installs' file
        self.assertFalse([x for x in self.m["entries"] if x["kind"] == "file" and x.get("path") == "/etc/modprobe.d/99-nvidia-greenboost.conf"])

    @unittest.skipUnless(
        os.path.exists(os.path.join(os.path.dirname(HERE), "manifest.json")) and
        json.load(open(os.path.join(os.path.dirname(HERE), "manifest.json")))["generated_from"]["sha256"] ==
        g.sha(open(os.path.join(REPO, "greenboost_setup.sh"), "rb").read()),
        "installer changed since manifest.json was generated (regenerate, then update the expected list)")
    def test_coverage_round_3_every_install_step_has_an_uninstall_step(self):
        # round 2 listed 14 leftovers; round 3 closed them (do_purge, patch 0006 + groupdel + ai-workstation.conf)
        out = io.StringIO()
        notes = json.load(open(os.path.join(os.path.dirname(HERE), "coverage_notes.json")))
        unc = g.cmd_coverage(self.m, out, notes)
        self.assertEqual({g.label(e) for e in unc}, set())
        left_notes = out.getvalue()
        self.assertNotIn("UNCOVERED", left_notes)

    def test_apply_then_uninstall_on_a_populated_fake_root(self):
        # files the installer edits but does not own already exist here
        self.put_root("etc/group", "root:x:0:\n")
        self.put_root("etc/systemd/system/ollama.service", '[Service]\nEnvironment="OLLAMA_KEEP_ALIVE=-1"\nExecStart=/usr/bin/ollama serve\n')
        self.put_root("etc/fstab", "UUID=x / ext4 defaults 0 1\n")
        self.put_root("etc/ld.so.preload", "/usr/local/lib/libgreenboost_audit.so\n/opt/other.so\n")
        before = snap(self.root)
        self.apply(self.m)
        c, _ = self.plan(self.m)
        self.assertEqual(set(c) - {"unchanged", "opaque"}, set())       # a second plan is empty
        # entries that depend on a run-time value are never half-applied
        self.assertIn('OLLAMA_KEEP_ALIVE=-1', self.read_root("etc/systemd/system/ollama.service"))
        self.assertEqual(self.read_root("etc/ld.so.preload"), "/opt/other.so\n")  # stale GreenBoost line scrubbed
        self.uninstall(self.m)
        after = snap(self.root)
        allow = {"etc/ld.so.preload"}                                   # a scrub is not undone
        self.assertEqual({k: v for k, v in before.items() if k not in allow}, {k: v for k, v in after.items() if k not in allow})

    def test_module_dir_flows_into_the_cli_wrapper(self):
        e = next(x for x in self.m["entries"] if x["id"] == "file:/usr/local/bin/greenboost")
        self.assertFalse(e["opaque"])
        self.assertIn(g.PH_MODULE.encode(), e["content"].encode())
        data = g.desired_bytes(e, REPO, {"MODULE_DIR": "/opt/gb"})
        self.assertIn(b'GB_SETUP="/opt/gb/greenboost_setup.sh"', data)
        self.assertNotIn(g.PH_MODULE.encode(), data)


if __name__ == "__main__":
    unittest.main()


class Round5Entries(unittest.TestCase):
    """SP62-02b: files and settings added by the swap and ollama-context patches."""
    @classmethod
    def setUpClass(cls):
        cls.m = json.load(open(os.path.join(REPO, "tools/manifest/manifest.json")))

    def test_ollama_ctx_conf_is_a_managed_file_entry(self):
        e = [x for x in self.m["entries"] if x.get("path") == "/etc/greenboost/ollama-ctx.conf"]
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0]["kind"], "file")

    def test_zz_model_slots_is_not_written_by_the_installer(self):
        self.assertFalse([x for x in self.m["entries"] if "zz-model-slots" in (x.get("path") or "")])

    def test_gb_swap_conf_is_owner_input_not_written(self):
        self.assertFalse([x for x in self.m["entries"] if x.get("kind") == "file" and x.get("path") == "/etc/greenboost/gb-swap.conf"])
