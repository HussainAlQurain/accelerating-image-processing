# Contributing to Real-Time Image Processing on GPUs

We welcome contributions from the community! Whether it's fixing a bug, improving performance, adding a new feature, or simply providing documentation updates, any contribution is appreciated. Below are a few guidelines to help you contribute effectively.

## How to Contribute

1. **Fork this repository**  
   Click the "Fork" button at the top of this page to create a personal copy on your GitHub account.

2. **Create a new branch**  
   Use a descriptive branch name (e.g., `feature-fast-gaussian`, `bugfix-sobel-edge`)  
   ```bash
   git checkout -b feature-fast-gaussian

3. Make your changes

- Ensure your code follows our existing style and structure.
- Write clear commit messages and document any significant changes in the code.

4. Test your changes

- If you modified existing code, ensure you run it with a sample video or input to confirm correctness and performance.
- If you added a new feature (e.g., new filter), provide a minimal example or test case.

5. Commit and push
```bash
git add .
git commit -m "Add faster Gaussian kernel for GPU"
git push origin feature-fast-gaussian
```
6. Open a Pull Request (PR)

- Go to your fork on GitHub, switch to your new branch, and click New Pull Request.
- In the PR description, explain what changes you made and why they’re beneficial.
- If applicable, attach logs, screenshots, or performance metrics.

## Reporting Issues
- If you encounter a bug, open an Issue and describe:
    1. Steps to reproduce the problem.
    2. Expected behavior vs. actual behavior.
    3. Any error messages or logs.
- Label the issue appropriately (e.g. bug, enhancement).

## Contact
hussain.qurain@outlook.com