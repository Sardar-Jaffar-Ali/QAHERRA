# Security and anonymity checklist

Before generating the anonymous review link, verify that the repository has:

- no API keys, tokens, passwords, or `.env` files;
- no author names, email addresses, affiliations, acknowledgements, or usernames;
- no local Windows user directories or private Google Drive paths;
- no notebook outputs or notebook execution metadata;
- no private Git remote URL in the distributed archive;
- no 3GPP documents or other content that cannot be redistributed;
- no multi-gigabyte embeddings, databases, or indexes.

Run from the repository root:

```bash
python scripts/security_scan.py
```

