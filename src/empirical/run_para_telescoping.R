# RUN ON UNI CLUSTER
rm(list = ls())
library(readxl)
library(future)
library(future.apply)
library(parallel)
source("functions_telescoping.R")
source("functions_ridge.R")

# Cali
set.seed(42)
n_sub <- 1000

cali_df <- readRDS("data/california_housing_openml.rds")
cali_df <- na.omit(cali_df)
cali_df <- cali_df[sample(seq_len(nrow(cali_df)), n_sub), ,drop = FALSE]
Y_cali <- as.numeric(cali_df$median_house_value) / 10000  #Scale target it is in 10,000$ units
X_cali <- cali_df[, names(cali_df) != "median_house_value", drop = FALSE]
X_cali <- model.matrix(~ 0 + ., data = X_cali)
X_cali <- as.matrix(X_cali)


#Concrete
d <- read_excel("data/Concrete_Data.xls")
colnames(d) <- c(
  "Cement", "Slag", "Fly ash", "Water", "concreteplasticizer",
  "Coarse aggregate", "Fine aggregate", "Age", "CCS")
Y_concrete <- d$CCS
X_concrete <- model.matrix(~ 0 + ., data = d[, names(d) != "CCS"])

# -----------------------------------------------------------------------------#
# Functions
# -----------------------------------------------------------------------------#

run_single_benchmark <- function(
    X, Y,
    split_seed,
    init_seed,
    hidden,
    lr,
    epochs,
    batch_size,
    lambda_grid,
    patience,
    tol,
    zero_init,
    scale_loss,
    train_frac,
    val_frac,
    artifact_policy) {
  
  # ---- DATA SPLITING -----
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
  
  # ---- TRAIN TELESCOPING MODEL -----
  model <- FitNN_Telescoping(
    X_train = X_train,
    Y_train = Y_train,
    X_val = X_val,
    Y_val = Y_val,
    X_test = X_test,
    hidden = hidden,
    lr = lr,
    epochs = epochs,
    tol = tol,
    batch_size = batch_size,
    zero_init = zero_init,
    patience = patience,
    scale_loss = scale_loss,
    seed = init_seed,
    show = TRUE
  )
  
  # ---- TELESCOPING PREDICTION  -----
  device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
  model$eval()
  
  X_test_scaled <- scale_test_data(X_test, model$scale_info)
  X_test_scaled_t <- torch_tensor(X_test_scaled, dtype = torch_float(), device = device)
  
  pred_test_tel <- as.numeric(model$S_test_tel %*% Y_train)
  with_no_grad({pred_test_fwd <- as.numeric(as_array(model(X_test_scaled_t)$detach()$cpu()))})
  
  smoothers <- list()
  
  if (isTRUE(artifact_policy$keep_smoothers)) {
    smoothers$S_tel_train <- model$S_train_tel
    smoothers$S_tel_test <- model$S_test_tel
  }
  
  list(
    config = list(
      init_seed = init_seed,
      split_seed = split_seed,
      hidden = hidden,
      lr = lr,
      epochs = epochs,
      batch_size = batch_size,
      zero_init = zero_init,
      scale_loss = scale_loss
    ),
    y_test = Y_test,
    pred_tel = pred_test_tel,
    pred_fwd = pred_test_fwd,
    embeddings = list(),
    smoothers = smoothers
  )
}

# -----------------------------------------------------------------------------#
# Benchmark grid
# -----------------------------------------------------------------------------#

benchmark_grid <- expand.grid(
  split_seed = 1:4,
  init_seed = 1:3,
  hidden = list(c(64, 64, 64)),
  lr = c(0.05),
  epochs = 200,
  batch_size = c(256),
  patience = 20,
  lambda_grid = list(10^seq(-6, 2, length.out = 10)),
  tol = 1e-4,
  train_frac = 0.6,
  val_frac = 0.2,
  zero_init = list(TRUE),
  scale_loss = list(TRUE)
)

artifact_policy <- list(
  keep_smoothers = FALSE
)

# -----------------------------------------------------------------------------#
# PARALLEL GPU SETUP
# -----------------------------------------------------------------------------#

n_workers <- 24
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

# -----------------------------------------------------------------------------#
# CALI
# -----------------------------------------------------------------------------#

timed <- Sys.time()

results_cali_tel <- future_lapply(
  seq_len(nrow(benchmark_grid)),
  function(i) {
    
    g <- benchmark_grid[i, ]
    
    run_single_benchmark(
      X = X_cali,
      Y = Y_cali,
      split_seed = g$split_seed,
      init_seed = g$init_seed,
      hidden = g$hidden[[1]],
      lr = g$lr,
      epochs = g$epochs,
      batch_size = g$batch_size,
      lambda_grid = g$lambda_grid[[1]],
      train_frac = g$train_frac,
      val_frac = g$val_frac,
      patience = g$patience,
      tol = g$tol,
      zero_init = g$zero_init[[1]],
      scale_loss = g$scale_loss[[1]],
      artifact_policy = artifact_policy
    )
  },
  future.seed = TRUE,
  future.packages = c("torch")
)

total_time <- as.numeric(difftime(Sys.time(), timed, units = "secs"))
cat("Cali time:", round(total_time / 60), "minutes\n")

output_file <- paste0(
  "/pfs/data6/home/tu/tu_tu/tu_zxoti46/thesis/results/results_cali_tel.rds"
)

saveRDS(results_cali_tel, file = output_file)

# -----------------------------------------------------------------------------#
#CONCRETE
# -----------------------------------------------------------------------------#

timed <- Sys.time()

results_concrete_tel <- future_lapply(
  seq_len(nrow(benchmark_grid)),
  function(i) {
    
    g <- benchmark_grid[i, ]
    
    run_single_benchmark(
      X = X_concrete,
      Y = Y_concrete,
      split_seed = g$split_seed,
      init_seed = g$init_seed,
      hidden = g$hidden[[1]],
      lr = g$lr,
      epochs = g$epochs,
      batch_size = g$batch_size,
      lambda_grid = g$lambda_grid[[1]],
      train_frac = g$train_frac,
      val_frac = g$val_frac,
      patience = g$patience,
      tol = g$tol,
      zero_init = g$zero_init[[1]],
      scale_loss = g$scale_loss[[1]],
      artifact_policy = artifact_policy
    )
  },
  future.seed = TRUE,
  future.packages = c("torch")
)

total_time <- as.numeric(difftime(Sys.time(), timed, units = "secs"))
cat("concreteconductor time:", round(total_time / 60), "minutes\n")

output_file <- paste0(
  "/pfs/data6/home/tu/tu_tu/tu_zxoti46/thesis/results/results_concrete_tel.rds"
)

saveRDS(results_concrete_tel, file = output_file)

plan(sequential)
parallel::stopCluster(cl)