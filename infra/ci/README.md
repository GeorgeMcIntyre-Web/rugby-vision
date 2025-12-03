# CI/CD Infrastructure

## Overview

This directory contains CI/CD configuration and scripts for Rugby Vision.

## GitHub Actions Workflows

### CI Workflow (`.github/workflows/ci.yml`)

Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**

1. **backend-tests**
   - Set up Python 3.11
   - Install dependencies
   - Lint with flake8
   - Format check with black
   - Type check with mypy
   - Run pytest with coverage
   - Upload coverage to Codecov

2. **frontend-tests**
   - Set up Node.js 18
   - Install dependencies
   - Lint with ESLint
   - Type check with TypeScript
   - Run tests (Jest)
   - Build production bundle

3. **docker-build**
   - Build Docker images for backend and frontend
   - Verify images build successfully
   - Use build cache for faster builds

### Deploy Workflow (`.github/workflows/deploy.yml`)

Manual workflow for deploying to staging or production.

**Trigger**: Manual workflow dispatch

**Jobs:**

1. **deploy**
   - Build and push Docker images to registry
   - Deploy to selected environment
   - Requires secrets: DOCKER_USERNAME, DOCKER_PASSWORD

## Local CI Testing

You can test CI steps locally before pushing:

### Backend

```bash
cd backend

# Lint
flake8 .

# Format
black .

# Type check
mypy .

# Tests
pytest --cov=. --cov-report=term-missing
```

### Frontend

```bash
cd frontend

# Lint
npm run lint

# Type check
npm run type-check

# Tests
npm test

# Build
npm run build
```

### Docker

```bash
# Build backend
docker build -t rugby-vision-backend:test ./backend

# Build frontend
docker build -t rugby-vision-frontend:test ./frontend

# Test with docker-compose
docker-compose up --build
```

## Required Secrets

Configure these secrets in GitHub repository settings:

- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password or access token
- Additional secrets for deployment (e.g., SSH keys, cloud credentials)

## Branch Protection

Recommended branch protection rules for `main`:

- Require pull request reviews
- Require status checks to pass:
  - backend-tests
  - frontend-tests
  - docker-build
- Require branches to be up to date
- Include administrators

## Deployment Environments

Configure environments in GitHub repository settings:

### Staging
- Auto-deploy on push to `develop` (optional)
- No approval required

### Production
- Manual deployment only
- Require approval from maintainers
- Deploy from `main` branch only

## Monitoring

- Build status badge: `[![CI](https://i.ytimg.com/vi/GlqQGLz6hfs/sddefault.jpg)`
- Coverage badge: Use Codecov badge

## Troubleshooting

### CI Fails on Type Check

- Review mypy errors
- Add type hints or type ignores as needed
- Initially set `continue-on-error: true` to not block PRs

### Docker Build Fails

- Check Dockerfile syntax
- Verify all dependencies are in requirements.txt / package.json
- Test build locally first

### Deployment Fails

- Check deployment logs
- Verify secrets are configured
- Test deployment script locally if possible
