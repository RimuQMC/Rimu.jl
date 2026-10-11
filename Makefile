.PHONY: changelog

# Regenerate the PR/issue links at the bottom of CHANGELOG.md.
# Run this after editing CHANGELOG.md and before committing.
changelog:
	julia --project=docs -e 'using Changelog; Changelog.generate(Changelog.CommonMark(), "CHANGELOG.md"; repo="RimuQMC/Rimu.jl")'
