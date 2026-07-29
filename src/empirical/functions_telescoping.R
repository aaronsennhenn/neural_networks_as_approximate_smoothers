#--------------------- TELESCOPING HELPER FUNCTIONS ---------------------#


flatten_grads <- function(grads) {
  torch_cat(lapply(grads, function(g) g$reshape(c(-1))))
}

zero_output_layer <- function(model) {
  with_no_grad({
    model$output$weight$zero_()
    model$output$bias$zero_()
  })
}

compute_jacobian <- function(model, X_t) {
  
  params <- model$parameters
  n <- as.integer(X_t$size()[1])
  p <- sum(vapply(params, function(z) as.numeric(z$numel()), numeric(1)))
  
  J <- torch_zeros(
    n, p,
    dtype = X_t$dtype,
    device = X_t$device
  )
  
  model$train()
  
  for (i in seq_len(n)) {
    model$zero_grad()
    
    out_i <- model(X_t[i, ]$unsqueeze(1))$squeeze()
    
    grads <- autograd_grad(
      outputs = out_i,
      inputs = params,
      retain_graph = FALSE,
      create_graph = FALSE,
      allow_unused = FALSE
    )
    
    J[i, ] <- flatten_grads(grads)
  }
  
  J
}

make_E_batch <- function(idx, n, device) {
  
  b <- length(idx)
  E <- matrix(0, nrow = b, ncol = n)
  E[cbind(seq_len(b), idx)] <- 1
  
  torch_tensor(E, dtype = torch_float(), device = device)
}

#--------------------------------------------------------------------#


#--------------------- TELESCOPING NN FUNCTIONS ---------------------#


FitNN_Telescoping <- function(X_train,
                              Y_train,
                              X_val = NULL,
                              Y_val = NULL,
                              X_test = NULL,
                              seed = 1,
                              hidden = c(20, 20),
                              lr = 0.01,
                              epochs = 100,
                              batch_size = 256,
                              patience = Inf,
                              tol = 0,
                              zero_init = TRUE,
                              show = TRUE,
                              scale_loss = TRUE) {
  
  torch_manual_seed(seed)
  model <- BuildNN(ncol(X_train), hidden)
  model_trained <- TrainNN_Telescoping(model, 
                                       X_train = X_train, 
                                       Y_train = Y_train,
                                       X_val = X_val,
                                       Y_val = Y_val,
                                       X_test = X_test,
                                       seed = seed,
                                       lr = lr,
                                       epochs = epochs,
                                       batch_size = batch_size,
                                       patience = patience,
                                       tol = tol,
                                       zero_init = zero_init,
                                       show = show,
                                       scale_loss = scale_loss)
  
  return(model_trained)
}


TrainNN_Telescoping <- function(model, 
                                X_train, 
                                Y_train,
                                X_test = NULL,
                                X_val = NULL,
                                Y_val = NULL,
                                seed = 1,
                                lr = 0.01,
                                epochs = 100,
                                batch_size = 256,
                                patience = Inf,
                                tol = 0,
                                zero_init = TRUE,
                                show = TRUE,
                                scale_loss = TRUE) {
  
  # ------------------
  # 1. Prepare data
  # ------------------
  X_train <- as.matrix(X_train)
  Y_train <- as.matrix(Y_train)
  X_val <- as.matrix(X_val)
  Y_val <- as.matrix(Y_val)
  n <- nrow(X_train)
  
  # Scale X 
  scale_info <- scale_train_data(X_train, Y_train)
  X_train_scaled <- scale_info$X_scaled
  X_val_scaled <- scale_test_data(X_val, scale_info)

  # Torch objects
  device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")
  model$to(device = device)
  X_t <- torch_tensor(X_train_scaled, dtype = torch_float(), device = device)
  Y_t <- torch_tensor(Y_train, dtype = torch_float(), device = device) #raw targets
  X_val_t <- torch_tensor(X_val_scaled, dtype = torch_float(), device = device)
  Y_val_t <- torch_tensor(Y_val, dtype = torch_float(), device = device) #raw targets
  
  ids_t <- torch_tensor(seq_len(n), dtype = torch_long(), device = device)
  
  # Data loader
  dataset <- tensor_dataset(X_t, Y_t, ids_t)
  loader <- dataloader(dataset, batch_size = batch_size, shuffle = TRUE)
  
  if (!is.null(X_test)) {
    X_test <- as.matrix(X_test)
    X_test_scaled <- scale_test_data(X_test, scale_info)
    n_test <- nrow(X_test_scaled)
    X_test_t <- torch_tensor(X_test_scaled, dtype = torch_float(), device = device)
    S_test <- torch_zeros(n_test, n, dtype = torch_float(), device = device)
  }
  
  
  
  # ------------------
  # 2. Training prep
  # ------------------
  S <- torch_zeros(n, n, dtype = torch_float(), device = device)
  
  best <- Inf
  wait <- 0
  n_batches <- length(loader)
  
  # Optimizer
  optimizer <- optim_sgd(model$parameters, lr = lr)
  
  # Zero init
  if (zero_init) { zero_output_layer(model) }
  
  # Loss scaling and factor
  continuous_target <- length(unique(Y_train)) > 2 
  sigma_y = scale_info$sigma_y
  
  loss_factor <- if (scale_loss && continuous_target) 1 / sigma_y^2 else 1
  
  loss_fn <- if (scale_loss && continuous_target) {
    function(preds, y) {torch_mean(torch_pow((preds - y) / sigma_y, 2))}
  } else {
    function(preds, y) {torch_mean(torch_pow(preds - y, 2))}
  }
  
  
  # ------------------
  # 3. Training loop
  # ------------------
  for (epoch in seq_len(epochs)) {
    
    total_loss <- 0
    
    coro::loop(for (batch in loader) {
      
      idx <- as.integer(as_array(batch[[3]]$cpu()))
      b <- length(idx)
      
      # Telescoping recursion
      J <- compute_jacobian(model, X_t)
      J_b <- J[idx, ]
      K_all_b <- torch_mm(J, J_b$t())
      E_b <- make_E_batch(idx, n, device)
      
      
      # Optional S_test update
      if (!is.null(X_test)) {
        J_test <- compute_jacobian(model, X_test_t)
        K_test_b <- torch_mm(J_test, J_b$t())
        
        S_test <- S_test + (2 * lr * loss_factor / b) * torch_mm(K_test_b, E_b - S[idx, ])
      }
      
      # Normal S update
      S <- S + (2 * lr * loss_factor / b) * torch_mm(K_all_b, E_b - S[idx, ])
      
      
      # Standard update
      optimizer$zero_grad()
      preds <- model(batch[[1]])
      loss <- loss_fn(preds, batch[[2]])
      loss$backward()
      optimizer$step()
      
      
      
      total_loss <- total_loss + loss$item()
      
    })
    
    avg_loss <- total_loss / n_batches
    
    with_no_grad({
      val_preds <- model(X_val_t)
      val_loss <- as.numeric(loss_fn(val_preds, Y_val_t))
    })
    
    # Early stopping
    print(val_loss)
    if (val_loss < best - tol) {
      best <- val_loss
      wait <- 0
      best_state <- model$state_dict()
      best_S <- S$clone()
      if (!is.null(X_test)) {
        best_S_test <- S_test$clone()
      }
    } else {
      wait <- wait + 1
    }
    
    if (show) {
      cat(sprintf("Epoch %d/%d - Validation Loss %.6f\n", epoch, epochs, val_loss))
    }
    
    if (wait >= patience) {
      if (show) cat("Early stopping.\n")
      break
    }
  }
  
  # ------------------
  # 6. Restore best fit
  # ------------------
  S <- best_S
  S_test <- best_S_test
  model$load_state_dict(best_state)
  
  
  # ------------------
  # 7. Final predictions
  # ------------------
  with_no_grad({
    yhat__train_fwd <- model(X_t)$detach()
    yhat_train_tel <- torch_mm(S, Y_t)
  })
  
  
  # ------------------
  # 8. Store results
  # ------------------
  model$S_train_tel <- as_array(S$cpu())
  model$S_test_tel <- as_array(S_test$cpu())
  model$yhat_train_fwd <- as.numeric(as_array(yhat__train_fwd$cpu()))
  model$yhat_train_tel <- as.numeric(as_array(yhat_train_tel$cpu()))
  model$scale_info <- scale_info
  
  return(model)
}


#--------------------------------------------------------------------#