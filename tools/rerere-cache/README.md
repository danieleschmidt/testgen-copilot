# Rerere Cache Audit

This directory contains Git rerere cache entries for auditing purposes.

## What is rerere?

Git rerere (reuse recorded resolution) records the resolution of merge conflicts and automatically applies the same resolution when the same conflict appears again.

## Audit Process

To review recorded conflict resolutions:

```bash
# Show recorded resolutions
git rerere diff

# List all recorded conflicts
git rerere status

# Clear all recorded resolutions (if needed)
git rerere clear
```

## Security Considerations

- Rerere only records conflict resolution patterns, not sensitive data
- All resolutions are applied automatically only for identical conflicts
- Manual review is still required for new conflict patterns
- This audit trail allows verification of automatic merge decisions

## Files in this directory

Cache entries from the .git/rr-cache directory are periodically copied here for version control and audit purposes.