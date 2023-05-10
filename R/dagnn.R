make_graph <- function(inputs, names) {
  
  dag_df <- purrr::map2_dfr(inputs, names,
                            ~ dplyr::tibble(from = .x, to = .y))
  dag_ig <- igraph::graph_from_data_frame(dag_df)
  
  dag_ig
  
}

expand_dots <- function(inputs, output_dims, names, ops) {
  
  output_dims <- output_dims <- purrr::map(output_dims,
                                           eval)
  dd <- stringr::str_detect(names, stringr::fixed("[.]"))
  
  if(any(dd)) {
    lens <- purrr::map_int(output_dims[dd],
                           length)
    new_names <- purrr::map2(names[dd], lens,
                             ~ glue::glue_data(.x = list(`.` = 1:.y),
                                          .x,
                                          .open = "[",
                                          .close = "]"))
    
    inputs_to_add <- purrr::map(new_names,
                                ~ c(list(character(0)), as.list(.x[-length(.x)])))
    
    new_inputs <- purrr::map2(inputs[dd], lens,
                              ~ expand_dots_inputs(.x, .y))

    names <- as.list(names)
    names[dd] <- new_names
    inputs[dd] <- new_inputs

    names <- purrr::flatten_chr(names)
    #inputs <- purrr::flatten(inputs)

    output_dims <- output_dims %>%
      purrr::flatten()
    # output_dims[dd] <- purrr::map2(output_dims[dd], lens,
    #                                ~ rep(.x, .y))
    # output_dims <- output_dims %>%
    #   purrr::flatten()

    inputs[dd] <- purrr::map2(inputs[dd], lens,
                                  ~ rep(list(.x), .y))
    
    inputs[dd] <- purrr::map2(inputs[dd], inputs_to_add,
                              ~ purrr::map2(.x, .y,
                                            function(x, y) c(x, y)))
    
    inputs <- inputs %>%
      purrr::list_flatten()
    
    ops[dd] <- purrr::map2(ops[dd], lens,
                           ~ rep(list(.x), .y))
    ops <- ops %>%
      purrr::flatten()

    list(names, inputs, output_dims, ops)
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
  ops[sapply(ops, is.null)] <- "" 
  list(inps, ops)
}

general_nn <- torch::nn_module("GeneralNN",
                               
                               initialize = function(names, output_dims, ops, inputs, fns, args, act) {
                                  input_names <- names[purrr::map_lgl(inputs,
                                                                      ~ length(.x) > 0)]
  
                                  input_dims <- purrr::map(inputs[input_names],
                                                           ~ output_dims[.x])
                                
                                  input_dims <- purrr::map(input_dims,
                                                           ~ sum(unlist(.x)))
                                  
                                  default_fn <- fns[[is.null(names(fns))]]
                                  default_act <- act[[is.null(names(act))]]
                                  if(length(args) > 0) {
                                    default_args <- args[[is.null(names(args))]]
                                  } else {
                                    default_args <- args
                                  }
                                  
                                  fns2 <- fns[input_names]
                                  fns2 <- purrr::map_if(fns2,
                                                        is.null,
                                                        ~ default_fn,
                                                        .else = ~.x)
                                  
                                  act2 <- act[input_names]
                                  act2 <- purrr::map_if(act2,
                                                        is.null,
                                                        ~ default_act,
                                                        .else = ~.x)
                                  
                                  args2 <- args[input_names]
                                  args2 <- purrr::map_if(args2,
                                                         is.null,
                                                         ~ default_args,
                                                         .else = ~.x)
                                  args_dims <- purrr::map2(input_dims, output_dims[input_names],
                                                           ~ list(.x, .y))
                                  
                                  args2 <- purrr::map2(args_dims, args2,
                                                       ~ c(.x, .y))
                                  
                                  self$layers <- purrr::map2(fns2, args2,
                                                        ~ rlang::exec(.x, !!!.y)) %>%
                                    setNames(input_names) %>%
                                    torch::nn_module_list()
                                  
                                  self$activations <- purrr::map(act2,
                                                                 ~ rlang::exec(.x)) %>%
                                    setNames(input_names) %>%
                                    torch::nn_module_list()
                                  
                              })

#' @importFrom stats setNames
make_forward <- function(names, inputs, ops, this_nn, terminal) {
  input_names <- names[purrr::map_lgl(inputs,
                                      ~ length(.x) > 0)]
  
  init_inputs <- names[!names %in% input_names]
  
  calls <- purrr::pmap(list(inputs[input_names], input_names, seq_along(input_names)), 
                       ~ list(rlang::expr(x <- torch::torch_cat(!!rlang::call2(rlang::expr(list), !!!rlang::syms(..1)),
                                                                2L)), 
                         rlang::expr(x <- self$layers[[!!..3]](x)),
                         rlang::expr(!!rlang::sym(..2) <- self$activations[[!!..3]](x))))
  
  if(length(terminal) > 1) {
    calls <- c(calls, rlang::expr(!!rlang::call2(rlang::expr(list), !!!rlang::syms(terminal))))
  } else {
    calls <- c(calls, rlang::expr(!!rlang::sym(terminal)))
  }
  
  bod <- rlang::expr({!!!purrr::list_flatten(calls)})
  
  
  general_nn_final <- torch::nn_module("GeneralNNFinalized",
                                       inherit = general_nn,
                                       forward = rlang::new_function(args = replicate(length(init_inputs), 
                                                                                      rlang::missing_arg()) %>%
                                                                       setNames(init_inputs) %>%
                                                                       rlang::pairlist2(!!!.),
                                                                     body = bod))
  
 general_nn_final
  
}

#' Create a neural network module by specifying a directed acyclic graph (DAG)
#' 
#' Using a list of formulas as input, `dagnn` generates an [nn_module()] 
#' neural network with a compact syntax
#'
#' @param ... A list of formulas specifying a directed acyclic graph (DAG). 
#' See details for the correct syntax.
#' @param .fns A list of functions for generating neural network layers.
#' Named elements are mapped to layer names in the DAG, a single unnamed element
#' can be included which acts as the default. `[.]` syntax is accepted for list names (see details).
#' @param .args A list of named argument lists. Each element should be a named list with
#' arguments to be passed to the corresponding `.fns`. In the upper level list, named elements 
#' are mapped to layer names in the DAG, and a single unnamed element can be included which 
#' acts as the default. `[.]` syntax is accepted for list names (see details).
#' @param .act A list of functions for generating neural network activation for corresponding layers.
#' Named elements are mapped to layer names in the DAG, a single unnamed element
#' can be included which acts as the default. `[.]` syntax is accepted for list names (see details). 
#'
#' @return A `dagnn` object, which subclasses a [nn_module()].
#' @export
#' 
#' @importFrom zeallot %<-%
#'
#' @examples
#' dag <- nndag(i_1 = ~ 19,
#'              c = ~ 1,
#'              p_1 = i_1 + c ~ 11,
#'             `e_[.]` = p_1 + c ~ c(32, 16, 8))
#' print(dag)
#' print(dag$layers)
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
  c(names, inputs, output_dims, ops) %<-% expand_dots(inputs, output_dims, names, ops)
  
  names(inputs) <- names
  names(output_dims) <- names
  names(ops) <- names
  
  dag_ig <- make_graph(inputs, names)
  dag_sorted <- names(igraph::topo_sort(dag_ig))
  dag_deg <- igraph::degree(dag_ig, mode = "out")
  terminal <- names(dag_deg)[dag_deg == 0]
  
  names <- dag_sorted
  inputs <- inputs[names]
  output_dims <- output_dims[names]
  ops <- ops[names]
  
  nn_make <- make_forward(names, inputs, ops, this_nn, terminal)
  
  this_nn <- nn_make(names, output_dims, ops, inputs, .fns, .args, .act)
  
  attr(this_nn, "dag") <- dag_ig
  attr(this_nn, "nn_module") <- nn_make
  
  class(this_nn) <- c("dagnn", class(this_nn))
  
  this_nn

  # list(inputs, ops, output_dims, names, dag_ig, this_nn, nn_make)
}