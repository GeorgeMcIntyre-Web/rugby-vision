# Deployment Infrastructure

## Overview

Deployment configurations and scripts for Rugby Vision.

## Deployment Options

### Option 1: Docker Compose (Simple)

**Best for**: Development, small-scale production

```bash
# On deployment server
git clone <repository-url>
cd rugby-vision
docker-compose up -d
```

**Pros**: Simple, works anywhere with Docker
**Cons**: No auto-scaling, manual management

### Option 2: Kubernetes (Scalable)

**Best for**: Production, multi-instance deployments

```bash
# Apply Kubernetes manifests
kubectl apply -f infra/deploy/k8s/
```

**Pros**: Auto-scaling, self-healing, load balancing
**Cons**: More complex setup

### Option 3: Cloud Platform (Managed)

**Options**:
- AWS: ECS, EKS, App Runner
- GCP: Cloud Run, GKE
- Azure: Container Instances, AKS

**Pros**: Managed infrastructure, easy scaling
**Cons**: Cloud vendor lock-in, costs

## Environment Variables

### Backend

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `DEBUG`: Enable debug mode (0 or 1)
- `ALLOWED_ORIGINS`: CORS allowed origins (comma-separated)

### Frontend

- `VITE_API_URL`: Backend API URL (build-time)
- `NODE_ENV`: Environment (development, production)

## Deployment Checklist

### Pre-Deployment

- [ ] All tests pass
- [ ] Code review approved
- [ ] Version tagged (semantic versioning)
- [ ] Changelog updated
- [ ] Environment variables configured
- [ ] Secrets stored securely

### Deployment Steps

1. **Build images**
   ```bash
   docker-compose build
   ```

2. **Tag images**
   ```bash
   docker tag rugby-vision-backend:latest your-registry/rugby-vision-backend:v1.0.0
   docker tag rugby-vision-frontend:latest your-registry/rugby-vision-frontend:v1.0.0
   ```

3. **Push images**
   ```bash
   docker push your-registry/rugby-vision-backend:v1.0.0
   docker push your-registry/rugby-vision-frontend:v1.0.0
   ```

4. **Deploy**
   - Docker Compose: `docker-compose pull && docker-compose up -d`
   - Kubernetes: `kubectl apply -f k8s/`
   - Cloud: Use platform-specific CLI

5. **Verify deployment**
   - Check health endpoints
   - Run smoke tests
   - Monitor logs for errors

### Post-Deployment

- [ ] Health checks passing
- [ ] Smoke tests pass
- [ ] Monitoring alerts configured
- [ ] Logs being collected
- [ ] Backup and rollback plan ready

## Rollback Procedure

If deployment fails:

1. **Immediate rollback**
   ```bash
   # Docker Compose
   docker-compose down
   docker-compose up -d  # Uses previous images

   # Kubernetes
   kubectl rollout undo deployment/rugby-vision-backend
   kubectl rollout undo deployment/rugby-vision-frontend
   ```

2. **Investigate issue**
   - Check logs: `docker-compose logs` or `kubectl logs`
   - Review recent changes
   - Test in staging environment

3. **Fix and redeploy**
   - Fix issue in code
   - Test thoroughly
   - Deploy again

## Monitoring

### Health Checks

- Backend: `http://backend:8000/health`
- Frontend: `http://frontend:80`

### Logging

**Docker Compose**:
```bash
# View logs
docker-compose logs -f

# View specific service
docker-compose logs -f backend
```

**Kubernetes**:
```bash
# View logs
kubectl logs -f deployment/rugby-vision-backend

# View all pods
kubectl logs -l app=rugby-vision --all-containers=true -f
```

### Metrics

Recommended metrics to track:

- Request rate (requests/second)
- Response time (p50, p95, p99)
- Error rate (%)
- CPU and memory usage
- Video processing latency

### Alerting

Set up alerts for:

- Service downtime (health check fails)
- High error rate (>5%)
- High latency (>10s for analysis)
- Resource exhaustion (CPU >80%, Memory >90%)

## Scaling

### Horizontal Scaling

**Docker Compose**:
```bash
docker-compose up -d --scale backend=3
```

**Kubernetes**:
```bash
kubectl scale deployment rugby-vision-backend --replicas=3
```

### Vertical Scaling

Update resource limits in:
- `docker-compose.yml` (resources section)
- `k8s/deployment.yaml` (resources section)

## Security

### Best Practices

- Use HTTPS in production (TLS certificates)
- Store secrets in environment variables or secret managers
- Regular security updates (base images, dependencies)
- Network isolation (firewall rules, VPC)
- Authentication and authorization (JWT, OAuth)

### Secrets Management

**Development**: `.env` file (gitignored)

**Production**:
- Docker: Docker secrets
- Kubernetes: Kubernetes secrets
- Cloud: AWS Secrets Manager, GCP Secret Manager, Azure Key Vault

## Backup and Recovery

### Data to Backup

- Application configuration
- User data (if any)
- Logs (for audit trail)

### Backup Schedule

- Configuration: On every change
- Data: Daily (if applicable)
- Logs: Continuous (to log aggregation service)

### Recovery Testing

- Test restore process monthly
- Document recovery steps
- Time recovery to ensure RTO is met

## Support

For deployment issues:

1. Check logs first
2. Review documentation
3. Contact DevOps team
4. Create issue in repository
