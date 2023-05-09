test_that("order of DAG formulas doesn't matter", {
  
  dag1 <- nndag(i_1 = ~ 19,
                c = ~ 1,
                p_1 = i_1 & c ~ 11,
                `e_[.]` = p_1 + c ~ c(32, 16, 8))
  
  dag2 <- nndag(i_1 = ~ 19,
                `e_[.]` = p_1 + c ~ c(32, 16, 8),
                p_1 = i_1 & c ~ 11,
                c = ~ 1)
  
  expect_true(igraph::isomorphic(attr(dag1, "dag"), attr(dag2, "dag")))
  
  
})
