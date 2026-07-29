#----------------------------NN FUNCTIONS---------------------------------#

FitNN <- function(X, Y,
                  X_val = NULL,
                  Y_val = NULL,
                  seed = 1,
                  hidden = c(16, 16, 16),
                  lr = 0.01,
                  epochs = 150,
                  batch_size = 256,
                  optimizer = "adam",
                  patience = 15,
                  tol = 0,
                  lambda_grid = 10^seq(-3, 2, length.out = 20),
                  n_ridge_cv_folds = 3,
                  ridge_cv = TRUE,
                  show = FALSE,
                  scale_loss = FALSE,
                  compute.oob.predictions = F) {
  
  torch_manual_seed(seed)
  model <- BuildNN(ncol(X), hidden)
  
  model_trained <- TrainNN(
    model, X, Y,
    X_val = X_val,
    Y_val = Y_val,
    seed = seed,
    lr = lr,
    epochs = epochs,
    batch_size = batch_size,
    optimizer = optimizer,
    patience = patience,
    tol = tol,
    show = show,
    scale_loss = scale_loss
  )
  
  model_trained$lambda_best = tune_ridge_lambda(X= X, 
                                               y= Y, 
                                               lambda_grid = lambda_grid, 
                                               n_folds = n_ridge_cv_folds, 
                                               seed= seed, 
                                               X_val = X_val, 
                                               y_val = Y_val, 
                                               cv = ridge_cv)$lambda_best
  
  return(model_trained)
}



BuildNN <- function(n_features, hidden) {
  
  net <- nn_module(
    "RegressionNN",
    
    initialize = function() {
      layer_sizes <- c(n_features, hidden)
      
      self$layers <- nn_module_list(
        lapply(1:(length(layer_sizes) - 1), function(i) {
          nn_linear(layer_sizes[i], layer_sizes[i + 1])
        })
      )
      
      # Output layer
      self$output <- nn_linear(hidden[length(hidden)], 1)
    },
    
    forward = function(x, return_last_hidden = FALSE) {
      #Forward through all hidden layers with ReLU
      for (i in 1:length(self$layers)) {
        x <- torch_relu(self$layers[[i]](x))
      }
      
      #Output layer (no ReLU here for regression)
      yhat <- self$output(x)
      
      if (return_last_hidden) {
        return(list(yhat = torch_squeeze(yhat), last_hidden = x))
      } else {
        return(yhat)
      }
    }
  )
  model <- net()
  return(model)
}

TrainNN <- function(model, X, Y,
                    X_val = NULL,
                    Y_val = NULL,
                    val_frac = 0.2,
                    seed = 1,
                    lr = 0.05,
                    epochs = 100,
                    batch_size = 256,
                    optimizer = "sgd",
                    patience = Inf,
                    tol = 0,
                    show = TRUE,
                    scale_loss = FALSE) {
  
  # 1. Prepare data
  X <- as.matrix(X)
  Y <- as.matrix(Y)
  
  n <- nrow(X)
  
  scale_info <- scale_train_data(X, Y)
  X_scaled <- scale_info$X_scaled
  
  # Torch objects
  device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
  model$to(device = device)
  
  X_t <- torch_tensor(X_scaled, dtype = torch_float(), device = device)
  Y_t <- torch_tensor(Y, dtype = torch_float(), device = device)
  
  dataset <- tensor_dataset(X_t, Y_t)
  loader <- dataloader(dataset, batch_size = batch_size, shuffle = TRUE)
  

  # 2. Training prep
  optimizer <- if (optimizer == "adam") {
    optim_adam(model$parameters, lr = lr)
  } else {
    optim_sgd(model$parameters, lr = lr)
  }
  
  continuous_target <- length(unique(Y)) > 2
  sigma_y <- scale_info$sigma_y
  
  loss_fn <- if (scale_loss && continuous_target) {
    function(preds, y) torch_mean(torch_pow((preds - y) / sigma_y, 2))
  } else {
    function(preds, y) torch_mean(torch_pow(preds - y, 2))
  }
  
  best <- Inf
  wait <- 0
  n_batches <- length(loader)
  
  # 3. Training loop
  for (epoch in seq_len(epochs)) {
    
    total_loss <- 0
    
    coro::loop(for (batch in loader) {
      
      optimizer$zero_grad()
      preds <- model(batch[[1]])
      loss <- loss_fn(preds, batch[[2]])
      loss$backward()
      optimizer$step()
      
      total_loss <- total_loss + loss$item()
    })
    
    avg_loss <- total_loss / n_batches
    
    
    # Early stopping on validation loss
    if (avg_loss < best - tol) {
      best <- avg_loss
      wait <- 0
      best_state <- model$state_dict()
    } else {
      wait <- wait + 1
    }
    
    if (wait >= patience) {
      if (show) cat("Early stopping.\n")
      break
    }
    if (show) {
      cat(sprintf("Epoch %d/%d - Loss %.6f\n", epoch, epochs, avg))
    }
  }
  

  # 4. Restore best fit
  model$load_state_dict(best_state)
  
  # 5. Store results
  model$scale_info <- scale_info
  
  return(model)
}

Predict_NN <- function(model, X_new, pred_type = "ridge") {
    
   #Predicting with dual solution
   W <- get_nn_weights(model, X_new)
   Y_train <- model$scale_info$Y_train
   pred_ridge <- as.numeric(W %*% Y_train)
    
   #Predicting with forward pass (doesn't produce valid smoother)
   model$eval()
   X_new <- scale_test_data(X_new, model$scale_info)
   X_t <- torch_tensor(as.matrix(X_new), dtype = torch_float())
   pred_normal <- as.numeric(model(X_t))
   
   if (pred_type == "normal"){
      output <- list(predictions = pred_normal, alt_prediction = pred_ridge)
   } else { 
      output <- list(predictions = pred_ridge, alt_prediction = pred_normal)
   }
   
  return(output)
}

#------------------------------------------------------------------------------#




#----------------------- WEIGHT COMPUTATION  ----------------------------------#

ExtractLastHiddenLayer <- function(model, X) {
  
  # Receives scaled X
  X <- as.matrix(X)
  device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
  model$eval()
  X_t <- torch_tensor(X, dtype = torch_float(), device = device)
  out <- model(X_t, return_last_hidden = TRUE)
  as.matrix(as_array(out$last_hidden$cpu()))
}


compute_omega <- function(PHI_Xj, PHI_X, lambda) {

  #Make sure embeddings are matrices
  PHI_X <- as.matrix(PHI_X)
  PHI_Xj <- as.matrix(PHI_Xj)
  
  ridge_inv <- solve(t(PHI_X) %*% PHI_X + lambda * diag(ncol(PHI_X)))
  W <- PHI_Xj %*% ridge_inv %*% t(PHI_X)
  
  return(W)  
}

get_nn_weights <- function(model, X_new){
  
  #Scale test data
  X_new <- as.matrix(X_new)
  X_new <- scale_test_data(X_new, model$scale_info)
  
  X_train <- model$scale_info$X_scaled
  lambda = model$lambda_best
  
  #Get embeddings and lambda to compute contributions
  test_embeddings <- ExtractLastHiddenLayer(model, X_new)
  train_embeddings <- ExtractLastHiddenLayer(model, X_train)
  
  W <- compute_omega(test_embeddings, train_embeddings, lambda)
  
  return(W)
}

#------------------------------------------------------------------------------#







#---------------------------RIDGE REGRESSION----------------------------#

tune_ridge_lambda <- function(X, y, lambda_grid, n_folds = 5, seed,
                              X_val = NULL, y_val = NULL, cv = TRUE) {
  
  X <- as.matrix(X)
  y <- as.numeric(y)
  
  # Standard validation
  if (!cv && !is.null(X_val) && !is.null(y_val)) {
    X_val <- as.matrix(X_val)
    y_val <- as.numeric(y_val)
    cv_errors <- numeric(length(lambda_grid))
    
    for (j in seq_along(lambda_grid)) {
      lambda <- lambda_grid[j]
      
      # Catch lambda values for which the inversion breaks
      cv_errors[j] <- tryCatch({
        beta <- solve(
          crossprod(X) + lambda * diag(ncol(X)),
          crossprod(X, y)
        )
        yhat_val <- as.numeric(X_val %*% beta)
        mean((y_val - yhat_val)^2)
      }, error = function(e) Inf)
    }
  } else {
    
    # Crossvalidation using only X
    set.seed(seed)
    fold_id <- sample(rep(1:n_folds, length.out = nrow(X)))
    
    cv_errors <- numeric(length(lambda_grid))
    
    for (j in seq_along(lambda_grid)) {
      lambda <- lambda_grid[j]
      fold_errors <- numeric(n_folds)
      
      for (k in 1:n_folds) {
        train_idx <- fold_id != k
        val_idx <- fold_id == k
        
        X_train <- X[train_idx, , drop = FALSE]
        y_train <- y[train_idx]
        X_val_fold <- X[val_idx, , drop = FALSE]
        y_val_fold <- y[val_idx]
        
        # Catch lambda values for which the inversion breaks
        fold_errors[k] <- tryCatch({
          beta <- solve(
            crossprod(X_train) + lambda * diag(ncol(X_train)),
            crossprod(X_train, y_train)
          )
          yhat_val <- as.numeric(X_val_fold %*% beta)
          mean((y_val_fold - yhat_val)^2)
          
        }, error = function(e) Inf)
      }
      
      cv_errors[j] <- mean(fold_errors)
    }
  }
  
  cv_results <- data.frame(
    lambda = lambda_grid,
    cv_error = cv_errors)
  lambda_best = lambda_grid[which.min(cv_errors)]
  
  return(list(lambda_best = lambda_best, cv_results = cv_results))
}
#------------------------------------------------------------------------------#



#-----------------------------DATA SCALING-------------------------------------#

scale_train_data <- function(X, Y) {
  
  # Identify 0/1 dummy variables
  is_dummy <- apply(
    X,
    2,
    function(z) all(z %in% c(0, 1))
  )
  
  mu_x <- apply(X, 2, mean)
  sigma_x <- apply(X, 2, sd)
  
  #Do not scale dummy variables
  mu_x[is_dummy] <- 0
  sigma_x[is_dummy] <- 1
  
  #Safety for any other constant columns
  sigma_x[!is.finite(sigma_x) | sigma_x == 0] <- 1
  
  X_scaled <- sweep(X, 2, mu_x, "-")
  X_scaled <- sweep(X_scaled, 2, sigma_x, "/")
  
  sigma_y <- sd(Y)
  mu_y <- mean(Y)
  Y_scaled <- (Y - mu_y) / sigma_y
  
  list(
    X_train = X,
    Y_train = Y,
    X_scaled = X_scaled,
    Y_scaled = Y_scaled,
    sigma_y = sigma_y,
    mu_y = mu_y,
    sigma_x = sigma_x,
    mu_x = mu_x,
    is_dummy = is_dummy
  )
}

scale_test_data <- function(X, scale) {
  
  X <- as.matrix(X)
  
  X_scaled <- sweep(X, 2, scale$mu_x, "-")
  X_scaled <- sweep(X_scaled, 2, scale$sigma_x, "/")
  
  X_scaled
}
#------------------------------------------------------------------------------#