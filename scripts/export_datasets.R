#!/usr/bin/env Rscript
# scripts/export_datasets.R

# Create data directory if it doesn't exist
data_dir <- "/Users/ramakrushnamishra/welsch-adenet/data"
if (!dir.exists(data_dir)) {
  dir.create(data_dir, recursive = TRUE)
}

# 1. Boston Housing Dataset (MASS)
library(MASS)
data(Boston, package = "MASS")
write.csv(Boston, file.path(data_dir, "Boston.csv"), row.names = FALSE)
cat("Exported Boston.csv\n")

# 2. hbk Dataset (robustbase)
library(robustbase)
data(hbk, package = "robustbase")
write.csv(hbk, file.path(data_dir, "hbk.csv"), row.names = FALSE)
cat("Exported hbk.csv\n")

# 3. NCI60 protein and doubling time (robustHD)
library(robustHD)
data(nci60, package = "robustHD")
nci_df <- as.data.frame(protein)
nci_df$DoublingTime <- cellLineInfo$DoublingTime
write.csv(nci_df, file.path(data_dir, "nci60_protein.csv"), row.names = FALSE)
cat("Exported nci60_protein.csv\n")
