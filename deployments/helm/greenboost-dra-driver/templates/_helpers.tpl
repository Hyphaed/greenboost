{{/*
Expand the name of the chart.
*/}}
{{- define "greenboost-dra-driver.name" - }}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "greenboost-dra-driver.fullname" - }}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "greenboost-dra-driver.chart" - }}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "greenboost-dra-driver.labels" - }}
helm.sh/chart: {{ include "greenboost-dra-driver.chart" . }}
{{ include "greenboost-dra-driver.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "greenboost-dra-driver.selectorLabels" - }}
app.kubernetes.io/name: {{ include "greenboost-dra-driver.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- if .componentName }}
app.kubernetes.io/component: {{ .componentName }}
{{- end }}
{{- end }}

{{/*
Template labels
*/}}
{{- define "greenboost-dra-driver.templateLabels" - }}
helm.sh/chart: {{ include "greenboost-dra-driver.chart" . }}
{{ include "greenboost-dra-driver.selectorLabels" . }}
{{- if .componentName }}
app.kubernetes.io/component: {{ .componentName }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "greenboost-dra-driver.serviceAccountName" - }}
{{- if .Values.serviceAccount.create }}
{{- default (include "greenboost-dra-driver.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Image reference
*/}}
{{- define "greenboost-dra-driver.fullimage" - }}
{{- .Values.image.repository }}:{{ .Values.image.tag }}
{{- end }}

{{/*
Namespace
*/}}
{{- define "greenboost-dra-driver.namespace" - }}
{{- default "greenboost-system" .Values.namespaceOverride }}
{{- end }}

{{/*
Ternary function
*/}}
{{- define "ternary" - }}
{{- if first . }}
{{- first . }}
{{- else }}
{{- last . }}
{{- end }}
{{- end }}