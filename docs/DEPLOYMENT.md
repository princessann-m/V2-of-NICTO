# Deployment

## Docker

```bash
docker build -f docker/Dockerfile.cpu -t nicto:cpu .
docker run -p 8000:8000 nicto:cpu
```

For GPU:

```bash
docker build -f docker/Dockerfile.gpu -t nicto:gpu .
docker run --gpus all -p 8000:8000 nicto:gpu
```

## API Server

```bash
uvicorn mom.api.server:app --host 0.0.0.0 --port 8000
```

## CLI

```bash
python -m mom.cli.main handle "Calculate 2+2"
python -m mom.cli.main benchmark --mode A
```

## Production Checklist

- Set `MOM_LLM_PROVIDER` to a real provider (openai, anthropic, etc.)
- Configure `MOM_LLM_API_KEY`
- Tune `MOM_DEADLINE` and `MOM_MAX_RETRIES`
- Enable `mixed_precision` and `distributed` for training
- Run benchmark before deploying: `python -m mom.cli.main benchmark --mode A`
