#!/usr/bin/env Rscript
# Run BayesSpace's official MCMC cluster() on the 25 datasets.
# Neighbours are built by KNN (RANN) because these tissues are not all
# integer grids; spatialCluster's grid-matching step is replaced while
# the posterior model is unchanged.
suppressMessages({
  library(BayesSpace)
  library(SingleCellExperiment)
  library(Matrix)
  library(irlba)
  library(RANN)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
data_dir <- args[1]
out_csv  <- args[2]
nrep <- if (length(args) >= 3) as.integer(args[3]) else 10000L
bs <- asNamespace("BayesSpace")
cluster <- get("cluster", envir = bs)
Mode <- get("Mode", envir = bs)
init_cluster <- get(".init_cluster", envir = bs)
manifest <- fromJSON(file.path(data_dir, "manifest.json"))
if (file.exists(out_csv)) {
  done <- read.csv(out_csv, stringsAsFactors = FALSE)$dataset
} else {
  done <- c()
}
for (i in seq_len(nrow(manifest))) {
  m <- manifest[i, ]
  if (m$name %in% done) {
    cat("skip", m$name, "\n")
    next
  }
  base <- file.path(data_dir, m$file)
  cat(sprintf("[%d/%d] %s n=%d K=%d ... ", i, nrow(manifest), m$name,
              m$n_spots, m$k_gt))
  lc <- readMM(paste0(base, "_logcounts.mtx"))     # genes x spots
  coords <- as.matrix(read.csv(paste0(base, "_coords.csv"), header = FALSE))
  d_pca <- min(15, nrow(lc) - 1)
  pca <- prcomp_irlba(t(as(lc, "CsparseMatrix")), n = d_pca,
                      center = TRUE, scale. = FALSE)
  Y <- pca$x
  n <- nrow(Y)
  # KNN neighbour list, 0-based indices (k = 4, ST-style neighbourhood)
  knn <- nn2(coords, k = 5)$nn.idx[, -1, drop = FALSE]
  df_j <- lapply(seq_len(n), function(r) as.integer(knn[r, ] - 1L))
  init <- init_cluster(Y, m$k_gt, init.method = "kmeans")
  t0 <- proc.time()
  results <- cluster(Y, m$k_gt, df_j, init = init, model = "t",
                     precision = "equal", mu0 = colMeans(Y),
                     lambda0 = diag(0.01, ncol(Y)), gamma = 2,
                     alpha = 1, beta = 0.01, nrep = nrep,
                     thin = max(1L, as.integer(nrep / 100)))
  elapsed <- (proc.time() - t0)["elapsed"]
  thin <- max(1L, as.integer(nrep / 100))
  burn.in <- as.integer(nrep / 10)
  .burn.in <- burn.in %/% thin
  .nrep <- nrep %/% thin
  zs <- results$z[seq(.burn.in + 2, .nrep + 1), ]
  if (.burn.in + 1 == .nrep) {
    labels <- matrix(zs, nrow = 1)
  } else {
    labels <- apply(zs, 2, Mode)
  }
  labels <- as.integer(unname(labels))
  write.csv(labels, paste0(base, "_bayesspace_pred.csv"), row.names = FALSE)
  row <- data.frame(dataset = m$name, category = m$category,
                    k_gt = m$k_gt, n_spots = m$n_spots,
                    n_predicted = length(unique(labels)),
                    runtime = as.numeric(elapsed))
  write.table(row, out_csv, sep = ",", append = file.exists(out_csv),
              quote = FALSE, col.names = !file.exists(out_csv), row.names = FALSE)
  cat("done in", round(elapsed, 1), "s, clusters =", length(unique(labels)), "\n")
}
cat("All finished\n")
