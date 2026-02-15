//! IO module for data loading and serialization
//!
//! This module provides dataset loaders for standard SLAM benchmarks
//! and utilities for reading/writing SLAM data.

mod dataset;

pub use dataset::{
    Dataset, DatasetConfig, DatasetError, DatasetIterator, DatasetMetadata, Frame, Result,
    TumRgbdDataset, KittiDataset, EurocDataset,
};
