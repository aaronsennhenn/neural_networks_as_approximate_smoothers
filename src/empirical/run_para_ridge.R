# RUN ON UNI CLUSTER
rm(list = ls())
library(future)
library(future.apply)
library(parallel)
library(readxl)
source("functions_ridge.R")

#Load data
set.seed(42)
n_sub <- 5000 #Subsample size

#Concrete
d <- read_excel("data/Concrete_Data.xls")
colnames(d) <- c(
  "Cement", "Slag", "Fly ash", "Water", "Superplasticizer",
  "Coarse aggregate", "Fine aggregate", "Age", "CCS")
Y_concrete <- d$CCS
X_concrete <- model.matrix(~ 0 + ., data = d[, names(d) != "CCS"])


cali_df <- readRDS("data/california_housing_openml.rds")
cali_df <- na.omit(cali_df)
cali_df <- cali_df[sample(seq_len(nrow(cali_df)), n_sub), , drop = FALSE]
Y_cali <- as.numeric(cali_df$median_house_value) / 10000 #scale target
X_cali <- cali_df[, names(cali_df) != "median_house_value", drop = FALSE]
X_cali <- model.matrix(~ 0 + ., data = X_cali)
X_cali <- as.matrix(X_cali)

# Superconductor
super_df <- readRDS("data/superconductivity_openml.rds")
super_df <- na.omit(super_df)
super_df <- super_df[sample(seq_len(nrow(super_df)), n_sub), , drop = FALSE]
Y_super <- as.numeric(super_df$critical_temp)
X_super <- super_df[, names(super_df) != "critical_temp", drop = FALSE]
X_super <- model.matrix(~ 0 + ., data = X_super)
X_super <- as.matrix(X_super)

run_single_benchmark_ridge <- function(
    X, Y,
    split_seed,
    init_seed,
    hidden,
    lr,
    epochs,
    batch_size,
    optimizer,
    lambda_grid,
    patience,
    tol,
    scale_loss,
    train_frac,
    val_frac,
    n_ridge_cv_folds,
    artifact_policy) {
  
  set.seed(split_seed)
  X <- as.matrix(X)
  Y <- as.numeric(Y)
  
  n <- nrow(X)
  n_train <- round(train_frac * n)
  n_val <- round(val_frac * n)
  
  idx <- sample(seq_len(n))
  train_idx <- idx[1:n_train]
  val_idx <- idx[(n_train + 1):(n_train + n_val)]
  test_idx <- idx[(n_train + n_val + 1):n]
  
  X_train <- X[train_idx, , drop = FALSE]
  Y_train <- Y[train_idx]
  X_val <- X[val_idx, , drop = FALSE]
  Y_val <- Y[val_idx]
  X_test <- X[test_idx, , drop = FALSE]
  Y_test <- Y[test_idx]
  
  model <- FitNN(
    X = X_train,
    Y = Y_train,
    X_val = X_val,
    Y_val = Y_val,
    hidden = hidden,
    optimizer = optimizer,
    lr = lr,
    epochs = epochs,
    batch_size = batch_size,
    patience = patience,
    tol = tol,
    scale_loss = scale_loss,
    seed = init_seed,
    show = FALSE
  )
  
  device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
  model$eval()
  
  X_train_scaled <- model$scale_info$X_scaled
  X_test_scaled <- scale_test_data(X_test, model$scale_info)
  X_val_scaled <- scale_test_data(X_val, model$scale_info)
  
  X_train_scaled_t <- torch_tensor(X_train_scaled, dtype = torch_float(), device = device)
  X_test_scaled_t <- torch_tensor(X_test_scaled, dtype = torch_float(), device = device)
  
  with_no_grad({
    yhat_train_fwd <- as.numeric(as_array(model(X_train_scaled_t)$detach()$cpu()))
    pred_test_fwd <- as.numeric(as_array(model(X_test_scaled_t)$detach()$cpu()))
  })
  
  Phi_train <- ExtractLastHiddenLayer(model, X_train_scaled)
  Phi_test <- ExtractLastHiddenLayer(model, X_test_scaled)
  Phi_val <- ExtractLastHiddenLayer(model, X_val_scaled)
  
  lambda_tune <- tune_ridge_lambda(
    X = Phi_train,
    y = Y_train,
    X_val = Phi_val,
    y_val = Y_val,
    lambda_grid = lambda_grid,
    n_folds = n_ridge_cv_folds,
    cv = FALSE,
    seed = split_seed
  )
  
  lambda_best <- lambda_tune$lambda_best
  cv_results <- lambda_tune$cv_results
  
  ridge_inv <- solve(crossprod(Phi_train) + lambda_best * diag(ncol(Phi_train)))
  beta_ridge <- ridge_inv %*% crossprod(Phi_train, Y_train)
  
  yhat_train_ridge <- as.numeric(Phi_train %*% beta_ridge)
  pred_test_ridge <- as.numeric(Phi_test %*% beta_ridge)
  
  embeddings <- list()
  smoothers <- list()
  
  if (isTRUE(artifact_policy$keep_embeddings)) {
    embeddings$Phi_train <- Phi_train
    embeddings$Phi_test <- Phi_test
  }
  
  if (isTRUE(artifact_policy$keep_smoothers)) {
    smoothers$S_train_ridge <- Phi_train %*% ridge_inv %*% t(Phi_train)
    smoothers$S_test_ridge <- Phi_test %*% ridge_inv %*% t(Phi_train)
  }
  
  list(
    config = list(
      init_seed = init_seed,
      split_seed = split_seed,
      n_ridge_cv_folds = n_ridge_cv_folds,
      hidden = hidden,
      lr = lr,
      epochs = epochs,
      batch_size = batch_size,
      optimizer = optimizer,
      scale_loss = scale_loss,
      lambda_best = lambda_best
    ),
    y_test = Y_test,
    pred_ridge = pred_test_ridge,
    pred_fwd = pred_test_fwd,
    ridge_cv = cv_results,
    embeddings = embeddings,
    smoothers = smoothers
  )
}

benchmark_grid <- expand.grid(
  split_seed = 1:4,
  init_seed = 1:3,
  n_ridge_cv_folds = 3,
  hidden = list(c(16, 16, 16), c(32, 32, 32), c(64, 64, 64)),
  lr = c(0.01, 0.05),
  batch_size = c(128, 256),
  optimizer = c("adam"),
  epochs = 150,
  patience = 15,
  lambda_grid = I(list(10^seq(-2, 2, length.out = 30))),
  tol = 1e-4,
  train_frac = 0.6,
  val_frac = 0.2,
  scale_loss = list(FALSE)
)

artifact_policy <- list(
  keep_embeddings = FALSE,
  keep_smoothers = FALSE
)


# PARALLEL SETUP
n_workers <- 32
cl <- parallel::makeCluster(n_workers, outfile = "")
worker_check <- parallel::clusterCall(
  cl = cl,
  fun = function() {
    list(
      pid = Sys.getpid()
    )
  }
)
print(worker_check)
plan(cluster, workers = cl)

timed <- Sys.time()

# CALI
timed <- Sys.time()
results_cali_ridge <- future_lapply(
  seq_len(nrow(benchmark_grid)),
  function(i) {
    
    g <- benchmark_grid[i, ]
    
    run_single_benchmark_ridge(
      X = X_cali,
      Y = Y_cali,
      split_seed = g$split_seed,
      init_seed = g$init_seed,
      n_ridge_cv_folds = g$n_ridge_cv_folds,
      hidden = g$hidden[[1]],
      lr = g$lr,
      epochs = g$epochs,
      batch_size = g$batch_size,
      optimizer = g$optimizer,
      lambda_grid = g$lambda_grid[[1]],
      train_frac = g$train_frac,
      val_frac = g$val_frac,
      patience = g$patience,
      tol = g$tol,
      scale_loss = g$scale_loss[[1]],
      artifact_policy = artifact_policy
    )
  },
  future.seed = TRUE,
  future.packages = c("torch")
)

total_time <- as.numeric(difftime(Sys.time(), timed, units = "secs"))
cat("Cali time:", round(total_time / 60), "minutes\n")
output_file <- paste0("/pfs/data6/home/tu/tu_tu/tu_zxoti46/thesis/results/results_cali_ridge.rds")
saveRDS(results_cali_ridge, file = output_file)

# SUPER
timed <- Sys.time()
results_super_ridge <- future_lapply(
  seq_len(nrow(benchmark_grid)),
  function(i) {
    
    g <- benchmark_grid[i, ]
    
    run_single_benchmark_ridge(
      X = X_super,
      Y = Y_super,
      split_seed = g$split_seed,
      init_seed = g$init_seed,
      hidden = g$hidden[[1]],
      n_ridge_cv_folds = g$n_ridge_cv_folds,
      lr = g$lr,
      epochs = g$epochs,
      batch_size = g$batch_size,
      optimizer = g$optimizer,
      lambda_grid = g$lambda_grid[[1]],
      train_frac = g$train_frac,
      val_frac = g$val_frac,
      patience = g$patience,
      tol = g$tol,
      scale_loss = g$scale_loss[[1]],
      artifact_policy = artifact_policy
    )
  },
  future.seed = TRUE,
  future.packages = c("torch")
)

total_time <- as.numeric(difftime(Sys.time(), timed, units = "secs"))
cat("Total time:", round(total_time / 60), "minutes")
output_file <- paste0("/pfs/data6/home/tu/tu_tu/tu_zxoti46/thesis/results/results_super_ridge.rds")
saveRDS(results_super_ridge, file = output_file)


# CONCRETE
timed <- Sys.time()
results_concrete_ridge <- future_lapply(
  seq_len(nrow(benchmark_grid)),
  function(i) {
    
    g <- benchmark_grid[i, ]
    
    run_single_benchmark_ridge(
      X = X_concrete,
      Y = Y_concrete,
      split_seed = g$split_seed,
      init_seed = g$init_seed,
      hidden = g$hidden[[1]],
      n_ridge_cv_folds = g$n_ridge_cv_folds,
      lr = g$lr,
      epochs = g$epochs,
      batch_size = g$batch_size,
      optimizer = g$optimizer,
      lambda_grid = g$lambda_grid[[1]],
      train_frac = g$train_frac,
      val_frac = g$val_frac,
      patience = g$patience,
      tol = g$tol,
      scale_loss = g$scale_loss[[1]],
      artifact_policy = artifact_policy
    )
  },
  future.seed = TRUE,
  future.packages = c("torch")
)

total_time <- as.numeric(difftime(Sys.time(), timed, units = "secs"))
cat("Total time:", round(total_time / 60), "minutes")
output_file <- paste0("/pfs/data6/home/tu/tu_tu/tu_zxoti46/thesis/results/results_concrete_ridge.rds")
saveRDS(results_concrete_ridge, file = output_file)








