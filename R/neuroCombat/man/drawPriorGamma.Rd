% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/viz.R
\name{drawPriorGamma}
\alias{drawPriorGamma}
\title{Draw prior distributions of location parameters (gamma estimates)}
\usage{
drawPriorGamma(combat.estimates, col = NULL, xlim = c(-3, 1.5), ylim = c(0, 3))
}
\arguments{
\item{combat.estimates}{estimates list from neuroCombat output}

\item{col}{Character or numeric vector specifying site colors. If NULL, default colors will be used.}

\item{xlim}{x-axis limits}

\item{ylim}{y-axis limits#'}
}
\value{
Nothing. A plot will be produced as side effect.
}
\description{
Draw prior distributions of location parameters (gamma estimates).
}
