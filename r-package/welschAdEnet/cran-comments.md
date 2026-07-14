# CRAN submission comments

## Test environments

* Ubuntu 24.04, R 4.3.3 (local `R CMD check --as-cran`)

## R CMD check results

0 errors | 0 warnings | 1 note

* "unable to verify current time" -- this note is an artifact of running
  the check in a network-isolated environment with no NTP access; it is
  not expected on a machine with normal internet access.

## Downstream dependencies

This is a new package; there are no downstream dependencies to check.
