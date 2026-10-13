"""One-time IBM Quantum account setup.

Run:  python scripts/setup_ibm.py

What it does: asks for your IBM Quantum Platform API key and instance (CRN), then
calls qiskit's `save_account`, which stores them in ~/.qiskit/qiskit-ibm.json:
your home folder, OUTSIDE this repository. The key is typed with hidden input,
never printed, never written to any project file, and never logged.

Where to find the values (IBM Quantum Platform, quantum.cloud.ibm.com):
  - API key: your profile/dashboard -> "API key" -> create one, copy it once.
  - Instance CRN: the "Instances" page -> your Open Plan instance -> copy its CRN
    (a long string starting with "crn:v1:bluemix:...").
"""

from getpass import getpass

from qiskit_ibm_runtime import QiskitRuntimeService

CHANNEL = "ibm_quantum_platform"  # verified against qiskit-ibm-runtime 0.50.0


def main() -> None:
    print("IBM Quantum account setup. Your key is saved to ~/.qiskit/, not to this repo.\n")
    token = getpass("API key (hidden as you type): ").strip()
    instance = input("Instance CRN (crn:v1:bluemix:...): ").strip()
    if not token or not instance.startswith("crn:"):
        raise SystemExit("Need both an API key and an instance CRN starting with 'crn:'. Nothing saved.")

    QiskitRuntimeService.save_account(
        channel=CHANNEL, token=token, instance=instance, overwrite=True, set_as_default=True
    )
    print("Saved. Checking the connection (read-only, uses no QPU time)...")
    service = QiskitRuntimeService()
    names = sorted(b.name for b in service.backends(simulator=False, operational=True))
    print(f"Connected. Quantum computers available to you: {', '.join(names) or 'none listed'}")


if __name__ == "__main__":
    main()
