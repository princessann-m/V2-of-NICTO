# NICTO Scaling Plan — Path to 1 Trillion Parameters

## Current Status

**Implementation**: Complete training infrastructure with real trainable models  
**Models**: All UNTRAINED — architectures ready, weights are random  
**Tests**: 247 passing  
**Hardware**: Not currently available for large-scale training  

---

## Scaling Roadmap

### Phase 1: Prototype Validation (0.1B total params)
**Goal**: Validate training pipeline end-to-end

| Model | Parameters | Count | Total |
|-------|------------|-------|-------|
| mamba_xsmall | 7M | 4 | 28M |
| top_model | 30M | 1 | 30M |
| verifier | 30M | 1 | 30M |
| judge | 30M | 2 | 60M |
| **Total** | | | **~148M** |

**Hardware**: 1-4 GPUs (RTX 3090/4090)  
**Data**: 1GB-10GB curated datasets  
**Time**: 1-2 weeks  
**Status**: Ready to begin

---

### Phase 2: Medium Scale (1B total params)
**Goal**: Demonstrate specialist expertise at medium scale

| Model | Parameters | Count | Total |
|-------|------------|-------|-------|
| mamba_small | 30M | 20 | 600M |
| mamba_medium | 130M | 3 | 390M |
| top_model | 100M | 1 | 100M |
| verifier | 130M | 1 | 130M |
| judge | 130M | 2 | 260M |
| **Total** | | | **~1.48B** |

**Hardware**: 16-32 GPUs (A100 40GB)  
**Data**: 100GB-1TB domain-specific datasets  
**Time**: 1-2 months  
**Status**: Infrastructure ready

---

### Phase 3: Large Scale (100B total params)
**Goal**: Demonstrate sparse expert routing at scale

| Model | Parameters | Count | Total |
|-------|------------|-------|-------|
| mamba_large | 370M | 50 | 18.5B |
| mamba_xl | 1B | 20 | 20B |
| mamba_xxl | 7B | 5 | 35B |
| top_model | 1B | 1 | 1B |
| verifier | 370M | 1 | 0.37B |
| judge | 370M | 2 | 0.74B |
| director | 370M | 1 | 0.37B |
| **Total** | | | **~75B** |

**Hardware**: 256-512 GPUs (A100 80GB)  
**Data**: 10TB+ diverse datasets  
**Time**: 6-12 months  
**Status**: Planning complete

---

### Phase 4: Full Scale (1T total params)
**Goal**: Full 310-expert pool with maximum capability

| Model | Parameters | Count | Total |
|-------|------------|-------|-------|
| mamba_xxl | 7B | 50 | 350B |
| mamba_xl | 1B | 100 | 100B |
| mamba_large | 370M | 160 | 59.2B |
| top_model | 1B | 1 | 1B |
| verifier | 370M | 1 | 0.37B |
| judge | 370M | 2 | 0.74B |
| director | 370M | 1 | 0.37B |
| router | 100M | 1 | 0.1B |
| **Total** | | | **~511B** |

**Note**: 1T requires either more experts at larger sizes or additional components (multimodal encoders, etc.)

**Hardware**: 1024+ GPUs (A100/H100 80GB)  
**Data**: 100TB+ curated datasets  
**Time**: 12-24 months  
**Status**: Architecture defined

---

## Training Order

### Recommended Sequence

1. **Foundation Models** (Months 1-3)
   - Train base Mamba (7B) on general text
   - Train encoder-decoder (1B) for task parsing
   - Validate with perplexity benchmarks

2. **First Experts** (Months 3-6)
   - Train 5-10 medium experts on key domains
   - Math, coding, reasoning, science, vision
   - Validate with domain benchmarks

3. **Distillation Pipeline** (Months 6-9)
   - Distill large experts into smaller ones
   - Create 50-100 total experts
   - Validate routing accuracy

4. **Verifier/Judge Training** (Months 9-12)
   - Train verifier on candidate quality data
   - Train judges on preference data
   - Validate agreement with human judgment

5. **Router Training** (Months 12-15)
   - Collect execution traces
   - Train learned router
   - Validate routing quality vs latency

6. **Scale Expansion** (Months 15-24)
   - Add more experts
   - Increase expert sizes
   - Implement multimodal components
   - Continuous evaluation and optimization

---

## Data Requirements by Phase

### Phase 1: Prototype
- Foundation: 1GB text
- Math: 10K problems
- Coding: 10K code pairs
- Total: ~1GB

### Phase 2: Medium Scale
- Foundation: 100GB text
- Math: 1M problems
- Coding: 1M code pairs
- Science: 500K Q&A pairs
- Total: ~1TB

### Phase 3: Large Scale
- Foundation: 10TB text
- Math: 100M problems
- Coding: 100M code pairs
- Science: 50M Q&A pairs
- Vision: 100M image-text pairs
- Total: ~10TB

### Phase 4: Full Scale
- Foundation: 100TB+ text
- All domains: Billions of samples
- Multimodal: Billions of image/video pairs
- Total: ~100TB+

---

## Hardware Estimates

### Phase 1: Prototype
- 1-4x RTX 3090/4090 (24GB)
- Cost: $5K-$20K
- Power: 1-2 kW

### Phase 2: Medium Scale
- 16-32x A100 40GB
- Cost: $200K-$500K
- Power: 10-20 kW

### Phase 3: Large Scale
- 256-512x A100 80GB
- Cost: $2M-$5M
- Power: 100-200 kW

### Phase 4: Full Scale
- 1024+ H100 80GB
- Cost: $10M-$30M
- Power: 500+ kW

---

## Cost Estimates

### Training Costs

| Phase | Compute | Data | Total Cost |
|-------|---------|------|------------|
| 1 | 1-4 GPUs, 2 weeks | 1GB | $500-$2K |
| 2 | 32 GPUs, 2 months | 1TB | $50K-$200K |
| 3 | 512 GPUs, 12 months | 10TB | $2M-$10M |
| 4 | 1024+ GPUs, 24 months | 100TB | $10M-$50M |

### Operational Costs (Monthly)

| Phase | GPUs | Power | Cloud | Total |
|-------|------|-------|-------|-------|
| 1 | 4 | $50 | $200 | $250 |
| 2 | 32 | $500 | $2K | $2.5K |
| 3 | 512 | $5K | $20K | $25K |
| 4 | 1024 | $20K | $80K | $100K |

---

## Risk Factors

1. **Hardware availability**: Large-scale training requires significant GPU resources
2. **Data quality**: Training data must be high-quality and diverse
3. **Training stability**: Large models can be unstable to train
4. **Cost**: Full 1T training requires millions of dollars
5. **Time**: Full training could take 1-2 years
6. **Expert collapse**: Router may overuse certain experts
7. **Scaling laws**: Unknown if sparse experts scale efficiently

---

## Mitigation Strategies

1. **Start small**: Validate with 7M-30M models first
2. **Incremental scaling**: Double model size only when smaller models work
3. **Distillation**: Use distillation to reduce compute needs
4. **Modular training**: Train components independently first
5. **Continuous evaluation**: Benchmark at every stage
6. **Fallback systems**: Maintain working smaller models while scaling

---

## Success Criteria

### Phase 1
- [ ] Models train without errors
- [ ] Loss decreases consistently
- [ ] Generated text is coherent
- [ ] Tests pass

### Phase 2
- [ ] Domain experts show specialization
- [ ] Router selects relevant experts
- [ ] Verification reduces hallucination
- [ ] Judges correlate with human judgment

### Phase 3
- [ ] 50+ experts trained and operational
- [ ] Routing accuracy >80%
- [ ] System completes complex tasks
- [ ] Latency <10s for simple tasks

### Phase 4
- [ ] 310 expert pool fully operational
- [ ] Total parameters ~1T
- [ ] State-of-the-art on benchmark tasks
- [ ] Production-ready inference

---

## Current Action Items

1. **Immediate**: Prepare foundation training data (100GB+)
2. **Week 1-2**: Train small models (7M-30M) to validate pipeline
3. **Week 3-4**: Set up medium-scale training environment
4. **Month 2**: Begin medium expert training
5. **Month 3**: Implement distillation pipeline
6. **Month 4**: Train first 50 experts
7. **Month 6**: Evaluate and optimize routing
8. **Month 12**: Scale to 100+ experts
9. **Month 24**: Full 1T parameter system (if resources available)

---

*This is a living document. Update as training progresses and hardware/data become available.*
