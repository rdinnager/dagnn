library(torch)

make_graph <- function(inputs, names, output_dims) {

}

expand_dots <- function(inputs, output_dims, names) {
  dd <- stringr::str_detect(names, stringr::fixed("[.]"))
  if(any(dd)) {
    lens <- purrr::map_int(output_dims[dd],
                           length)
    new_names <- purrr::map2(names[dd], lens,
                             ~ glue::glue_data(.x = list(`.` = 1:.y),
                                          .x,
                                          .open = "[",
                                          .close = "]"))
    new_inputs <- purrr::map2(inputs[dd], lens,
                              ~ expand_dots_inputs(.x, .y))

    names <- as.list(names)
    names[dd] <- new_names
    inputs[dd] <- new_inputs

    names <- purrr::flatten_chr(names)
    #inputs <- purrr::flatten(inputs)

    output_dims <- purrr::map(output_dims,
                              eval) %>%
      purrr::flatten()
    # output_dims[dd] <- purrr::map2(output_dims[dd], lens,
    #                                ~ rep(.x, .y))
    # output_dims <- output_dims %>%
    #   purrr::flatten()

    inputs[dd] <- purrr::map2(inputs[dd], lens,
                                  ~ rep(list(.x), .y))
    inputs <- inputs %>%
      purrr::flatten()

    list(names, inputs, output_dims)
  }

}

expand_dots_inputs <- function(inputs, lens) {
  new_inputs <- purrr::map(inputs,
                           ~ glue::glue_data(.x = list(`.` = 1:lens),
                                             .x,
                                             .open = "[",
                                             .close = "]"))
  purrr::flatten_chr(new_inputs)
}

get_inputs <- function(inputs) {
  inps <- purrr::map(inputs,
                     all.vars)
  ops <- purrr::map_if(inputs,
                       ~ !is.null(.x),
                       rlang::call_name,
                       .else = NULL)
  list(inps, ops)
}

nndag <- function(..., .fns = list(torch::nn_linear),
                  .args = list(),
                  .act = list(torch::nn_relu)) {
  dots <- rlang::enquos(...)
  not_forms <- !purrr::map_lgl(dots, rlang::quo_is_call)
  rlang::env_bind(rlang::current_env(), !!!dots[not_forms])
  forms <- purrr::map(dots[!not_forms], rlang::quo_squash)
  inputs <- purrr::map(forms, rlang::f_lhs)
  output_dims <- purrr::map(forms, rlang::f_rhs)
  names <- names(forms)

  c(inputs, ops) %<-% get_inputs(inputs)
  c(names, inputs, output_dims) %<-% expand_dots(inputs, output_dims, names)

  list(inputs, ops, output_dims, names)
}

tt<- nndag(i_1 = ~ 19,
           c = ~ 1,
           p_1 = i_1 & c ~ 11,
           `e_[.]` = p_1 + c ~ c(32, 16, 8))

tt


