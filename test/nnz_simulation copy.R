library(tidyverse)

setwd("/home/maximilian/Documents/repositories/ProximalSEM.jl/test/")

# read errors -------------------------------------------------------------

res_l0 <- read_csv("res_l0.csv")
res_l1 <- read_csv("res_l1.csv")
res_ml <- read_csv("res_ml.csv")

res_l0$reg <- "l0"
res_l1$reg <- "l1"
res_ml$reg <- "ml"

res <- bind_rows(
  res_l0,
  res_l1,
  res_ml
)

res <- rename(res,
  a = α,
  which_a = which_α
)

# convergence -------------------------------------------------------------

res %>% 
  group_by(reg) %>% 
  summarise(converged = sum(converged))

res %>% 
  group_by(reg) %>% 
  summarise(converged = sum(is.infinite(bic)))

res %>% 
  ggplot(aes(x = a)) +
  facet_wrap(~reg) +
  geom_histogram(bins = 10)


# recover structure -------------------------------------------------------

res <- mutate(res, 
  recovered = zero_c + nonzero_ic,
  )

res %>% 
  filter(reg %in% c("l0", "l1")) %>% 
  ggplot(aes(x = recovered)) +
  geom_bar() +
  facet_wrap(~reg)

# cor: 9
# in: 2*9
# -> 27

res %>% 
  filter(reg %in% c("l0", "l1")) %>% 
  ggplot(aes(x = zero_c)) +
  geom_bar() +
  facet_wrap(~reg)

res %>% 
  filter(reg %in% c("l0", "l1")) %>% 
  ggplot(aes(x = nonzero_ic)) +
  geom_bar() +
  facet_wrap(~reg)


# error -------------------------------------------------------------------

res <- pivot_longer(
  res,
  starts_with("error"),
  values_to = "error",
  names_to = "type")

res %>% group_by(reg, type) %>% summarise(error_max = max(error))
cut = 1

res %>% 
  filter(error < cut) %>% 
  ggplot(aes(x = error, color = reg)) +
  geom_density() +
  facet_grid(cols = vars(type)) +
  theme_minimal()

res %>% 
  group_by(reg, type) %>% 
  summarise(error_max = quantile(error, 0.75))

