"""
Utility functions for computing DSM and semantic segmentation metrics.
This module is standalone and can be used independently.

Author: Ahmad Naghavi, OzU 2024
"""

import numpy as np
from typing import Tuple


def r2_score(y, yhat, eps=1e-8):
    """
    Compute the coefficient of determination (R²) for regression tasks.
    
    R² represents the proportion of variance in the dependent variable that is 
    predictable from the independent variable(s). It ranges from -∞ to 1, where:
    - 1.0 indicates perfect prediction
    - 0.0 indicates the model performs no better than a horizontal line at the mean
    - Negative values indicate the model performs worse than the mean
    
    Args:
        y (np.ndarray): Ground truth values (1D or 2D array)
        yhat (np.ndarray): Predicted values (1D or 2D array)
        eps (float): Small epsilon value to avoid division by zero
        
    Returns:
        float: R² score computed across all pixels, or None if input is empty
    """
    # Flatten arrays to ensure 1D computation across all pixels
    y_flat = y.flatten()
    yhat_flat = yhat.flatten()
    
    if y_flat.size == 0:
        return None
    
    # Residual sum of squares
    ss_res = np.sum((y_flat - yhat_flat) ** 2)
    
    # Total sum of squares
    ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
    
    # Handle edge case where all ground truth values are the same
    if ss_tot < eps:
        return 0.0
    
    return 1.0 - ss_res / (ss_tot + eps)


def compute_dsm_metrics(
    verbose,
    logger,
    total_delta1,
    total_delta2,
    total_delta3,
    total_mse,
    total_mae,
    total_rmse,
    total_rmse_building,
    total_high_rise_rmse,
    total_mid_rise_rmse,
    total_low_rise_rmse,
    count_high_rise,
    count_mid_rise,
    count_low_rise,
    dsm_tile,
    dsm_pred,
    gt_mask=None,
    pred_mask=None,
    total_r2=0.0,
    low_rise_max=15.0,
    mid_rise_max=40.0
):
    """
    Compute Digital Surface Model (DSM) evaluation metrics for a single tile.
    
    Args:
        verbose (bool): If True, logs detailed metrics for each tile
        logger: Logger object for output messages
        total_delta1 (float): Running total for delta1 accuracy metric
        total_delta2 (float): Running total for delta2 accuracy metric
        total_delta3 (float): Running total for delta3 accuracy metric
        total_mse (float): Running total for Mean Squared Error
        total_mae (float): Running total for Mean Absolute Error 
        total_rmse (float): Running total for Root Mean Squared Error
        total_rmse_building (float): Running total for RMSE on building pixels only
        total_high_rise_rmse (float): Running total for RMSE on high-rise buildings
        total_mid_rise_rmse (float): Running total for RMSE on mid-rise buildings
        total_low_rise_rmse (float): Running total for RMSE on low-rise buildings
        count_high_rise (int): Count of tiles with high-rise buildings
        count_mid_rise (int): Count of tiles with mid-rise buildings
        count_low_rise (int): Count of tiles with low-rise buildings
        dsm_tile (numpy.ndarray): Ground truth DSM tile
        dsm_pred (numpy.ndarray): Predicted DSM tile
        gt_mask (numpy.ndarray, optional): Ground truth segmentation mask (1 for buildings)
        pred_mask (numpy.ndarray, optional): Predicted segmentation mask (1 for buildings)
        total_r2 (float): Running total for R² score on all pixels
        low_rise_max (float): Maximum height for low-rise buildings (default: 15.0m)
        mid_rise_max (float): Maximum height for mid-rise buildings (default: 40.0m)
        
    Returns:
        tuple: Updated totals for all metrics
    """
    # Ensure both arrays have the same shape
    assert dsm_tile.shape == dsm_pred.shape, (
        f"Shape mismatch: dsm_tile shape {dsm_tile.shape}, "
        f"dsm_pred shape {dsm_pred.shape}"
    )

    # Copy to avoid modifying the originals
    dsm_tile_ = dsm_tile.copy()
    dsm_pred_ = dsm_pred.copy()

    # Replace zero or negative values to avoid division by zero or invalid ratios
    eps = 1e-5
    dsm_pred_[dsm_pred_ <= 0] = eps
    dsm_tile_[dsm_tile_ <= 0] = eps

    # Flatten arrays for computation
    dsm_tile_ = dsm_tile_.flatten()
    dsm_pred_ = dsm_pred_.flatten()
 
    # Compute error metrics: MSE, MAE, RMSE
    abs_diff = np.abs(dsm_pred_ - dsm_tile_)
    tile_mse = np.mean(abs_diff ** 2)
    tile_mae = np.mean(abs_diff)
    tile_rmse = np.sqrt(tile_mse)

    # Compute delta metrics
    max_ratio = np.maximum(dsm_pred_ / dsm_tile_, dsm_tile_ / dsm_pred_)
    tile_delta1 = np.mean(max_ratio < 1.25)
    tile_delta2 = np.mean(max_ratio < 1.25 ** 2)
    tile_delta3 = np.mean(max_ratio < 1.25 ** 3)

    # Compute R² score for overall DSM prediction
    tile_r2 = r2_score(dsm_tile_, dsm_pred_)
    if tile_r2 is None:
        tile_r2 = 0.0

    # Compute building-specific height metrics
    tile_rmse_building = 0.0
    tile_high_rise_rmse = None
    tile_mid_rise_rmse = None
    tile_low_rise_rmse = None
    
    if gt_mask is not None:
        # Ensure GT mask has the same shape as DSM tiles
        if gt_mask.shape != dsm_tile.shape[:2]:
            if verbose:
                logger.warning(f"GT mask shape mismatch: gt_mask {gt_mask.shape}, DSM shape {dsm_tile.shape[:2]}")
        else:
            # Building pixels are where mask == 1
            building_mask_gt = (gt_mask == 1).flatten()
            
            # Calculate RMSE for building pixels only (based on ground truth mask)
            if np.sum(building_mask_gt) > 0:
                dsm_pred_buildings = dsm_pred_[building_mask_gt]
                dsm_tile_buildings = dsm_tile_[building_mask_gt]
                tile_rmse_building = np.sqrt(np.mean((dsm_pred_buildings - dsm_tile_buildings) ** 2))
            
            # Calculate height-category-specific RMSE based on GT building heights
            # Combine building mask with height thresholds
            low_rise_mask = building_mask_gt & (dsm_tile_ >= 1) & (dsm_tile_ < low_rise_max)
            mid_rise_mask = building_mask_gt & (dsm_tile_ >= low_rise_max) & (dsm_tile_ < mid_rise_max)
            high_rise_mask = building_mask_gt & (dsm_tile_ >= mid_rise_max)
            
            # Low-rise buildings RMSE
            if np.sum(low_rise_mask) > 0:
                low_rise_pred = dsm_pred_[low_rise_mask]
                low_rise_gt = dsm_tile_[low_rise_mask]
                tile_low_rise_rmse = np.sqrt(np.mean((low_rise_pred - low_rise_gt) ** 2))
                count_low_rise += 1
            
            # Mid-rise buildings RMSE
            if np.sum(mid_rise_mask) > 0:
                mid_rise_pred = dsm_pred_[mid_rise_mask]
                mid_rise_gt = dsm_tile_[mid_rise_mask]
                tile_mid_rise_rmse = np.sqrt(np.mean((mid_rise_pred - mid_rise_gt) ** 2))
                count_mid_rise += 1
            
            # High-rise buildings RMSE
            if np.sum(high_rise_mask) > 0:
                high_rise_pred = dsm_pred_[high_rise_mask]
                high_rise_gt = dsm_tile_[high_rise_mask]
                tile_high_rise_rmse = np.sqrt(np.mean((high_rise_pred - high_rise_gt) ** 2))
                count_high_rise += 1

        if verbose:
            logger.info(f"Tile MSE   : {tile_mse:.4f}")
            logger.info(f"Tile MAE   : {tile_mae:.4f}")
            logger.info(f"Tile RMSE  : {tile_rmse:.4f}")
            logger.info(f"Tile R^2    : {tile_r2:.4f}")
            logger.info(f"Tile Delta1: {tile_delta1:.4f}")
            logger.info(f"Tile Delta2: {tile_delta2:.4f}")
            logger.info(f"Tile Delta3: {tile_delta3:.4f}")
            if gt_mask is not None:
                logger.info(f"Tile RMSE Building: {tile_rmse_building:.4f}")
                if tile_low_rise_rmse is not None:
                    logger.info(f"Tile Low-rise RMSE: {tile_low_rise_rmse:.4f}")
                if tile_mid_rise_rmse is not None:
                    logger.info(f"Tile Mid-rise RMSE: {tile_mid_rise_rmse:.4f}")
                if tile_high_rise_rmse is not None:
                    logger.info(f"Tile High-rise RMSE: {tile_high_rise_rmse:.4f}")

    # Update running totals
    total_mse  += tile_mse
    total_mae  += tile_mae
    total_rmse += tile_rmse
    total_r2 += tile_r2

    total_delta1 += tile_delta1
    total_delta2 += tile_delta2
    total_delta3 += tile_delta3
    
    # Update building-specific totals
    total_rmse_building += tile_rmse_building
    
    if tile_high_rise_rmse is not None:
        total_high_rise_rmse += tile_high_rise_rmse
    if tile_mid_rise_rmse is not None:
        total_mid_rise_rmse += tile_mid_rise_rmse
    if tile_low_rise_rmse is not None:
        total_low_rise_rmse += tile_low_rise_rmse

    # Return updated totals + filtered arrays
    return (
        total_delta1,
        total_delta2,
        total_delta3,
        total_mse,
        total_mae,
        total_rmse,
        total_rmse_building,
        total_high_rise_rmse,
        total_mid_rise_rmse,
        total_low_rise_rmse,
        count_high_rise,
        count_mid_rise,
        count_low_rise,
        total_r2,
        dsm_tile_,
        dsm_pred_
    )
