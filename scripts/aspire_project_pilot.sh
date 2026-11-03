#!/bin/bash
#PBS -N vs-proj-pilot
#PBS -q normal
#PBS -l select=1:ngpus=1
#PBS -l walltime=06:00:00
#PBS -j oe
set -euo pipefail
if [ -z "${PBS_JOBID:-}" ]; then
  echo "Submit with scripts/submit_aspire_project_pilot.sh; computation requires PBS." >&2
  exit 2
fi
cd "${VS_ASPIRE_REPO:-${PBS_O_WORKDIR:?PBS working directory missing}}"
export VS_ASPIRE_WORKFLOW=project-pilot
# Reuse the fit launcher's tested PATH, shared Git/Python and branch checks.
exec /bin/bash scripts/aspire_resource_trial.sh
