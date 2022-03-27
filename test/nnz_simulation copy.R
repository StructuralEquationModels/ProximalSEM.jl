library(tidyverse)

setwd("/home/maximilian/Documents/repositories/ProximalSEM.jl/test/")

# read errors -------------------------------------------------------------

res <- read_csv("res.csv")
res <- rename(res,
  a_l0_cv = α_l0_cv,
  a_l1_cv = α_l1_cv,
  a_l0_bic = α_l0_bic,
  a_l1_bic = α_l1_bic
)

res %>% 
  pivot_longer(
    starts_with("a_"),
    names_to = "type",
    values_to = "alpha"
  ) %>% 
  ggplot(aes(x = alpha)) +
  geom_bar() + 
  facet_wrap(~type)

res %>% 
  select(starts_with("converged")) %>% 
  colSums()

res <- mutate(res, 
  l0_recovered_bic = zero_c_bic + nonzero_ic_bic,
  l0_recovered_cv = zero_c_cv + nonzero_ic_cv,
  )

# cor: 9
# in: 2*9
# -> 27

res %>% 
  ggplot(aes(x = l0_recovered_bic)) +
  geom_bar()

res %>% 
  ggplot(aes(x = zero_c_bic)) +
  geom_bar()

res %>% 
  ggplot(aes(x = nonzero_ic_bic)) +
  geom_bar()

res <- res %>% pivot_longer(
  starts_with("error"),
  values_to = "error",
  names_to = "type"
)

get_reg <- function(string){
  if(str_detect(string, "l0")){
    return("l0")
  }else if(str_detect(string, "l1")){
    return("l1")
  }else{
    return("ml")
  }
}

res <- res %>% 
  mutate(
    pars = ifelse(str_detect(type, "_ic_"), "incorrect", "correct"),
    reg = map_chr(type, get_reg),
    criterion = ifelse(str_detect(type, "bic"), "bic", "cv")
  )

er_cutoff = 1

res %>% 
  filter(error < er_cutoff, reg %in% c("l0", "l1")) %>% 
  ggplot(aes(x = error, color = reg)) +
  geom_density() +
  facet_grid(rows = vars(pars), cols = vars(criterion))

res %>% 
  group_by(reg, pars, criterion) %>% 
  summarize(error = max(error[error != Inf]))

res %>% 
  filter(error < 1, reg %in% c("l0", "l1")) %>% 
  ggplot(aes(x = error, color = criterion)) +
  geom_density() +
  facet_grid(rows = vars(pars), cols = vars(reg)) +
  theme_minimal()

res %>% 
  filter(error < er_cutoff, ((criterion == "bic")|(reg == "ml"))) %>% 
  ggplot(aes(x = error, color = reg)) +
  geom_density() +
  facet_grid(rows = vars(pars)) +
  theme_minimal()
