# Contributing to localmcplib

Thank you for your interest in contributing to `localmcplib`.

## Governance

This is a Salesforce-sponsored project. Salesforce employees serve as project
administrators and have final responsibility for deciding which contributions
are accepted.

## Issues and proposals

Use [GitHub Issues](https://github.com/salesforce-misc/localmcplib/issues) to
report bugs, request enhancements, or discuss ideas. Before starting a
substantial change, open an issue so the approach can be discussed with the
maintainers.

Security vulnerabilities must not be reported in a public issue. Follow the
instructions in [SECURITY.md](SECURITY.md) instead.

## Development

Install the complete development environment:

```console
make install
```

Run all repository checks before submitting a pull request:

```console
make ci
make build
```

`make test`, `make coverage`, and `make ci` detect the host OS and require its
real sandbox enforcement: Seatbelt on macOS and bubblewrap on Linux. To run
only a backend's enforcement tests, use `make test-seatbelt` or
`make test-bubblewrap`; these fail when that backend is unavailable.

Changes to observable behavior should include focused tests at a stable public
boundary. Keep pull requests narrowly scoped, document public contract changes,
and update examples when appropriate.

## Pull requests

1. Fork the repository and create a branch for your change.
2. Make the change and add or update tests and documentation.
3. Run the repository checks listed above.
4. Open a pull request against `main` and link any relevant issues.
5. Sign the Salesforce Contributor License Agreement when prompted.

All changes require peer review. By contributing code, you agree to license
your contribution under the terms of [LICENSE.txt](LICENSE.txt) and to sign the
[Salesforce CLA](https://cla.salesforce.com/sign-cla). Contributors must also
follow the project [Code of Conduct](CODE_OF_CONDUCT.md).
