wheatdata <- wheatdata %>%
  arrange(Year) %>%
  # *** NEW: Calculate Weight based on position/year ***
  # The weight is 1.0 for the most recent observation and decreases linearly.
  # We use the row index after sorting (the lowest index is the oldest year).
  mutate(
    Weight_Index = row_number(),
    Max_Weight = max(Weight_Index),
    # Calculate a simple linear weight where older years have lower weights
    Weight = Weight_Index / Max_Weight 
  )

# Create the Lagged_Production and Lagged_Cost columns
wheatdata <- wheatdata %>%
  mutate(
    Lagged_Production = lag(Production, n = 1, default = NA),
    Lagged_Cost = lag(Cost, n = 1, default = NA) # <-- NEW LAG FOR COST
  )

wheatdata <- wheatdata %>%
  mutate(
    # New: Create a running average of Production over the last 3 years
    Lagged_Avg_Prod_3yr = (lag(Production, 1) + lag(Production, 2) + lag(Production, 3)) / 3,
    
    # New: Create a running average of Cost over the last 3 years
    Lagged_Avg_Cost_3yr = (lag(Cost, 1) + lag(Cost, 2) + lag(Cost, 3)) / 3
  )

wheatdata <- wheatdata %>%
  mutate(
    # Interaction 1: Temp and Precipitation often interact to affect yield
    Temp_Precip_Int = Temp * Precipitation,
    # Interaction 2: Economic factors might interact with the long-term trend
    Prod_Cost_Int = Lagged_Production * Lagged_Cost 
  )

wheatdata <- wheatdata %>%
  mutate(
    # Squared Temperature: To model the optimal temperature point
    Temp_Sq = Temp * Temp,
    # Squared Humidity: To model the optimal humidity point
    Humidity_Sq = Humidity * Humidity
  )
# Remove the first row (the oldest year) because it now has NA for Lagged variables.
wheatdata <- na.omit(wheatdata)

# --- 2. Chronological Data Split ---

split_point <- floor(0.6 * nrow(wheatdata))

train_data <- wheatdata[1:split_point, ]
test_data <- wheatdata[(split_point + 1):nrow(wheatdata), ]

# --- 3. Prepare Training Data for M5P Model ---

# We now include Lagged_Production and Lagged_Cost, but exclude current Cost to avoid collinearity.
train_data_for_model <- train_data %>% 
  select(
    Lagged_Production, 
    Lagged_Cost, # <-- ADDED LAGGED COST AS PREDICTOR
    # --- Other Causal Factors ---
    Humidity, 
    Precipitation, 
    Temp,
    Temp_Precip_Int, # <--- NEW
    Prod_Cost_Int,   # <--- NEW
    Temp_Sq,         # <--- NEW
    Humidity_Sq,
    Lagged_Avg_Prod_3yr, Lagged_Avg_Cost_3yr,
    # REMOVED: Year, Sunshine, current Cost (based on leakage/collinearity findings)
    -Year, -Cost, -Sunshine # Exclude unwanted/redundant columns
  )

train_weights <- train_data$Weight
train_target <- train_data$Production

my_tune_grid <- expand.grid(
  pruned = c(TRUE), 
  smoothed = c(TRUE),
  rules=FALSE
)

# --- 4. Train the M5P Model (using weights and current tune settings) ---

m5p_model_weighted <- caret::train(
  x=train_data_for_model,
  y=train_target,
  weights=train_weights,
  method="M5",
  trControl=trainControl(method="none"),
  tuneGrid = my_tune_grid
)

# --- 5. Prepare Test Data for Prediction ---

test_data_for_prediction <- test_data %>% 
  select(
    Lagged_Production, 
    Lagged_Cost, # <-- ADDED LAGGED COST FOR PREDICTION
    Humidity, 
    Precipitation, 
    Temp,
    Temp_Precip_Int, # <--- NEW
    Prod_Cost_Int,   # <--- NEW
    Temp_Sq,         # <--- NEW
    Humidity_Sq,Lagged_Avg_Prod_3yr, Lagged_Avg_Cost_3yr,
    # Exclude unwanted/redundant columns
    -Year, -Cost, -Sunshine
  )

predictions_weighted <- predict(m5p_model_weighted, newdata=test_data_for_prediction)

# --- 6. Calculate Metrics (Metrics code remains correct) ---

actual_values_test <- test_data$Production
mae <- mean(abs(predictions_weighted-actual_values_test), na.rm=TRUE)
mae
rmse <- sqrt(mean((predictions_weighted -actual_values_test)^2, na.rm=TRUE))
rmse
actual_mean_test <- mean(actual_values_test, na.rm = TRUE)
ssr <- sum((predictions_weighted - actual_values_test)^2, na.rm = TRUE)
sst <- sum((actual_values_test - actual_mean_test)^2, na.rm = TRUE)
r_squared <- 1 - (ssr / sst)
r_squared
