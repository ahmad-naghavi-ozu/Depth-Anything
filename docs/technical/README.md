# Technical Documentation

This directory contains technical documentation for the Depth Anything adaptations and extensions.

## Remote Sensing (RS) Height Estimation

- **[README_RS.md](./README_RS.md)** - Comprehensive guide for RS height estimation adaptation
- **[RS_CONFIG_GUIDE.md](./RS_CONFIG_GUIDE.md)** - Detailed configuration parameters and setup guide

## Overview

The Remote Sensing adaptation allows the Depth Anything model to process satellite/aerial RGB imagery and produce height maps (Digital Surface Models) for building height estimation and urban analysis applications.

### Quick Navigation

- **Getting Started**: See [README_RS.md](./README_RS.md#quick-start)
- **Dataset Setup**: See [RS_CONFIG_GUIDE.md](./RS_CONFIG_GUIDE.md#dataset-configuration)
- **Training**: See [README_RS.md](./README_RS.md#2-training)
- **Inference**: See [README_RS.md](./README_RS.md#3-inference)
- **Troubleshooting**: See [RS_CONFIG_GUIDE.md](./RS_CONFIG_GUIDE.md#troubleshooting-common-issues)

### Key Features

- **Minimal Pipeline Changes**: Core architecture (DINOv2 + DPT) unchanged
- **Height-Specific Processing**: Proper handling of DSM data and height scaling  
- **Flexible Data Formats**: Supports PNG, JPEG, TIFF for both RGB and height maps
- **Complete Workflow**: From dataset validation to inference
- **Production Ready**: Comprehensive error handling and validation

### Architecture

```
RS RGB Image (H×W×3) → DINOv2 Encoder → DPT Decoder → Height Map (H×W×1)
```

The adaptation treats height estimation as depth estimation with different data interpretation, maintaining compatibility with the original Depth Anything pipeline while adding RS-specific functionality.