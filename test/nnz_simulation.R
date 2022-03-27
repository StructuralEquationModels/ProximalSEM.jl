library(tidyverse)

setwd("/home/maximilian/Documents/repositories/ProximalSEM.jl/test/")

# read errors -------------------------------------------------------------

error_incorrect_ml <- read_csv("error_incorrect_ml.csv")
error_incorrect_l0 <- read_csv("error_incorrect_l0.csv")
error_incorrect_l1 <- read_csv("error_incorrect_l1.csv")

error_correct_ml <- read_csv("error_correct_ml.csv")
error_correct_l0 <- read_csv("error_correct_l0.csv")
error_correct_l1 <- read_csv("error_correct_l1.csv")


error_incorrect_ml$type <- "incorrect"
error_incorrect_l0$type <- "incorrect"
error_incorrect_l1$type <- "incorrect"

error_correct_ml$type <- "correct"
error_correct_l0$type <- "correct"
error_correct_l1$type <- "correct"

error_incorrect_ml$reg <- "ml"
error_incorrect_l0$reg <- "l0"
error_incorrect_l1$reg <- "l1"

error_correct_ml$reg <- "ml"
error_correct_l0$reg <- "l0"
error_correct_l1$reg <- "l1"

error <- bind_rows(
  error_incorrect_ml,
  error_incorrect_l0,
  error_incorrect_l1,
  error_correct_ml,
  error_correct_l0,
  error_correct_l1
)

error <- error %>% 
  pivot_longer(
    starts_with("x"), 
    names_to = "alpha", 
    values_to = "error")

alpha_values <- c(seq(0.01, 0.1, 0.01), seq(0.2, 1.5, 0.1))
names(alpha_values) <- str_c("x", 1:24)

error$alpha <- recode(error$alpha, !!!alpha_values)

# read minimum alphas -----------------------------------------------------
α_l0_best <- read_csv("α_l0_best.csv")
α_l1_best <- read_csv("α_l1_best.csv")

# plot all errors ---------------------------------------------------------

error %>% 
  ggplot(aes(x = alpha, y = error, color = interaction(type, reg))) +
  geom_point()

error %>% filter(alpha > 0.1) %>% 
  ggplot(aes(x = alpha, y = error, color = interaction(type, reg))) +
  geom_point()

error %>% filter(alpha > 0.1, type == "incorrect") %>% 
  ggplot(aes(x = alpha, y = error, color = reg)) +
  geom_point()

error %>% filter(alpha > 0.1, type == "correct") %>% 
  ggplot(aes(x = alpha, y = error, color = reg)) +
  geom_point()

error_sum <- error %>% 
  filter(alpha > 0.1) %>% 
  group_by(type, reg, alpha) %>% 
  summarise(mean_error = mean(error, trim = 0.05),
            median_error = median(error))

error_sum %>% filter(type == "correct") %>% 
  ggplot(aes(x = alpha, y = mean_error, color = reg)) +
  geom_point() + geom_line()

error_sum %>% filter(type == "incorrect") %>% 
  ggplot(aes(x = alpha, y = mean_error, color = reg)) +
  geom_point() + geom_line()

error_sum %>% filter(type == "correct") %>% 
  ggplot(aes(x = alpha, y = median_error, color = reg)) +
  geom_point() + geom_line()

error_sum %>% filter(type == "incorrect") %>% 
  ggplot(aes(x = alpha, y = median_error, color = reg)) +
  geom_point() + geom_line()

error %>% 
  filter(reg == "l0", type == "incorrect", error < 100) %>%
  ggplot(aes(x = error)) +
  geom_density() +
  facet_wrap(~alpha)

α_l0_best <- 
  mutate(α_l0_best, alpha_value = alpha_values[α_l0_best$which_alpha])

α_l1_best <- 
  mutate(α_l1_best, alpha_value = alpha_values[α_l1_best$which_alpha])

α_l0_best %>% 
  ggplot(aes(x = alpha_value)) +
  geom_bar()

α_l1_best %>% 
  ggplot(aes(x = alpha_value)) +
  geom_bar()

error$data_id <- rep(str_c("data", 1:100), 3*2*24)

α_l0_best$data_id <- str_c("data", 1:100)
α_l1_best$data_id <- str_c("data", 1:100)

α_l0_best$reg <- "l0"
α_l1_best$reg <- "l1"

α_l0_best <- mutate(
    α_l0_best, 
    id = str_c(reg, alpha_value, data_id))

α_l1_best <- mutate(
  α_l1_best, 
  id = str_c(reg, alpha_value, data_id))

best_trials = c(α_l1_best$id, α_l0_best$id)

error <- error %>% 
  mutate(id = str_c(reg, alpha, data_id))

error2 <- filter(error, id %in% best_trials)

error2 %>% filter(type == "correct", error < 1) %>% 
  ggplot(aes(x = error, color = reg)) + geom_density()

error2 %>% filter(type == "incorrect", error < 1) %>% 
  ggplot(aes(x = error, color = reg)) + geom_density()


error2 %>% 
  pivot_wider(id_cols = c(id),
              names_from = c(reg, type),
              values_from = error)
