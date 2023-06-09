% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/dagnn.R
\name{nndag}
\alias{nndag}
\title{Create a neural network module by specifying a directed acyclic graph (DAG)}
\usage{
nndag(
  ...,
  .fns = list(torch::nn_linear),
  .args = list(),
  .act = list(torch::nn_relu)
)
}
\arguments{
\item{...}{A list of formulas specifying a directed acyclic graph (DAG).
See details for the correct syntax.}

\item{.fns}{A list of functions for generating neural network layers.
Named elements are mapped to layer names in the DAG, a single unnamed element
can be included which acts as the default. \verb{[.]} syntax is accepted for list names (see details).}

\item{.args}{A list of named argument lists. Each element should be a named list with
arguments to be passed to the corresponding \code{.fns}. In the upper level list, named elements
are mapped to layer names in the DAG, and a single unnamed element can be included which
acts as the default. \verb{[.]} syntax is accepted for list names (see details).}

\item{.act}{A list of functions for generating neural network activation for corresponding layers.
Named elements are mapped to layer names in the DAG, a single unnamed element
can be included which acts as the default. \verb{[.]} syntax is accepted for list names (see details).}
}
\value{
A \code{dagnn} object, which subclasses a \code{\link[=nn_module]{nn_module()}}.
}
\description{
Using a list of formulas as input, \code{dagnn} generates an \code{\link[=nn_module]{nn_module()}}
neural network with a compact syntax
}
\examples{
dag <- nndag(i_1 = ~ 19,
             c = ~ 1,
             p_1 = i_1 + c ~ 11,
            `e_[.]` = p_1 + c ~ c(32, 16, 8))
print(dag)
print(dag$layers)
}
