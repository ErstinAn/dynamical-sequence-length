# Environment record

The cleanup used Python 3.12.14 on Linux.
The dependency file records the installed runtime packages.
It does not contain packages that only serve other research projects.

| Component | Recorded version |
| --- | --- |
| PyTorch | `2.15.0.dev20260830+cu130` |
| TorchTitan | `737594746fda65a6d94dc9482ef07863a80c8588` |
| torchdata | `0.11.0` |
| torchao | `0.19.0.dev20260831+cu130` |
| datasets | `4.7.0` |
| tokenizers | `0.23.1` |
| NumPy | `2.5.2` |
| PyArrow | `25.0.1` |
| PyYAML | `6.0.3` |

The package uses private TorchTitan model functions and private FSDP state fields.
A version change can change their behavior.
The version file is a record of the test environment.
It does not prove that every wheel remains available from its package server.

The installation test uses the existing runtime dependencies in a separate environment.
A complete installation from external package servers remains untested.
The previous optimizer research repository is not necessary for this package.

## Installation with an existing runtime

Use this procedure with an environment that contains the recorded runtime packages.

1. Check the installed versions against `requirements-lock.txt`.
2. Install this package.

   ```bash
   python -m pip install --no-deps -e .
   ```

3. Run the CPU tests in section 6 of the [run guide](getting_started.md).
4. Run the GPU smoke in section 7 of the run guide.

## Version changes

1. Install the new versions in a separate environment.
2. Run the CPU tests.
3. Run the distributed tests.
4. Run the GPU tests.
5. Compare loss, gradient norm, and optimizer state with the reference results.
6. Record the new versions only after the tests pass.
