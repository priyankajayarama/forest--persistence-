# Contributing to Forest Persistence Segmentation

Thank you for your interest in contributing! This guide explains how to get involved.

---

## Ways to contribute

- Report bugs via GitHub Issues
- Suggest new features or improvements
- Improve documentation or add examples
- Submit code via Pull Requests
- Share the project and give feedback

---

## Getting started

### 1. Fork and clone

```bash
git clone https://github.com/YOUR_USERNAME/forest-segmentation.git
cd forest-segmentation
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Create a feature branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## Making changes

### Code style
- Follow PEP 8
- Use descriptive variable names
- Add a docstring to every function
- Keep functions short and focused (one responsibility)

### Commit messages
Use clear, present-tense messages:
```
Add multi-class forest type segmentation
Fix overflow in synthetic tile generation
Update README with deployment instructions
```

### Tests
Add or update tests for any code you change:
```bash
pytest tests/ -v
```

All tests must pass before submitting a PR.

---

## Submitting a Pull Request

1. Push your branch to your fork
2. Open a PR against the `main` branch
3. Fill in the PR template:
   - What does this PR do?
   - How was it tested?
   - Any screenshots or outputs?
4. Wait for review — we aim to respond within 48 hours

---

## Reporting bugs

Open a GitHub Issue with:
- A clear title
- Steps to reproduce
- Expected vs actual behaviour
- Your Python version and OS
- Any error messages or tracebacks

---

## Code of conduct

Be respectful, constructive, and inclusive. Harassment of any kind will not be tolerated.
