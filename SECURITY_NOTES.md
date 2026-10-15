# Dependency security scope

Sentence-transformers must be at least 5.6.0 for the local-model loading fix in GHSA-jhr6-gm9c-rqjv. The wrapper explicitly keeps `trust_remote_code=False`; its regression test uses an in-memory model stub and downloads no model. Pytest must be at least 9.0.3 for GHSA-6w46-j5rx-g56g.

GitHub also reports unpatched ChromaDB advisories, including [GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf). The reported critical attack requires a Chroma HTTP server, collection-update permissions and malicious remote model configuration. This repository constructs only local EphemeralClient/PersistentClient instances; it does not start or connect to a Chroma HTTP service. Do not expose a Chroma server or accept untrusted collection/model configuration as part of this tool's deployment. Existing alerts remain visible; this update does not claim ChromaDB is patched or suppress its advisories.
