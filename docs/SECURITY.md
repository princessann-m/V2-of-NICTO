Security and sandboxing guidance
================================

This project includes scripts to train models. Training and code execution can be risky
if not sandboxed. Follow these guidelines before running training jobs:

- Isolate training hosts from sensitive networks and data.
- Use non-root users inside containers; avoid mounting sensitive host directories.
- Limit resources with cgroups and container runtime options (CPU, memory, disk, network).
- Validate all datasets and code before training. Do not run untrusted preprocessing scripts.
- Use private networks and IAM roles for any cloud access; do not embed credentials in code.

Sandbox checklist:
- Run training within containers (`docker/Dockerfile.gpu`), or better, managed clusters (Kubernetes with PodSecurityPolicies).
- Ensure model artifacts are stored in controlled storage, with access logging.
- Consider using a dedicated training VM with no SSH access except through jump box.
