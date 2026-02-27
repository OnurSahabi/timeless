#' Parallel ARIMA Model Selection (forecast backend)
#'
#' Performs parallel grid search over ARIMA(p,d,q) models using
#' \code{forecast::Arima} and selects the best model according to
#' AIC, BIC, or AICc after Ljung-Box residual diagnostic filtering.
#'
#' @param data Numeric, ts, or zoo series.
#' @param max_p Maximum AR order.
#' @param max_q Maximum MA order.
#' @param max_lag Lag length for Ljung-Box test.
#' @param workers Number of parallel workers.
#' @param criterion Selection criterion: "AIC", "BIC", or "AICc".
#' @param include_mean Logical. Include mean in ARIMA model.
#'
#' @return A fitted \code{forecast::Arima} model.
#'
#' @importFrom forecast Arima ndiffs
#' @importFrom parallel makeCluster stopCluster parLapply clusterEvalQ clusterExport
#' @importFrom stats Box.test
#' @export
#'
#' @examples
#' \dontrun{
#' model <- autoparima(
#'   data = rnorm(500),
#'   max_p = 4,
#'   max_q = 4,
#'   max_lag = 20,
#'   workers = 2,
#'   criterion = "AIC"
#' )
#' summary(model)
#' }

autoparima <- function(data,
                       max_p,
                       max_q,
                       max_lag,
                       workers = 4,
                       criterion = c("AIC","BIC","AICc"),
                       include_mean = TRUE) {

  if(missing(data) || missing(max_p) || missing(max_q) || missing(max_lag)) {
    stop("Please provide values for: data, max_p, max_q, and max_lag.")
  }

  criterion <- match.arg(criterion)

  d <- forecast::ndiffs(data)
  grid <- expand.grid(p = 0:max_p, q = 0:max_q)

  cl <- parallel::makeCluster(workers)
  on.exit(parallel::stopCluster(cl), add = TRUE)

  parallel::clusterEvalQ(cl, library(forecast))
  parallel::clusterExport(
    cl,
    varlist = c("data","d","grid","max_lag","include_mean"),
    envir = environment()
  )

  results <- parallel::parLapply(cl, seq_len(nrow(grid)), function(i){

    p <- grid$p[i]
    q <- grid$q[i]

    tryCatch({

      model <- forecast::Arima(
        data,
        order = c(p, d, q),
        include.mean = include_mean
      )

      lb <- stats::Box.test(
        residuals(model),
        lag = max_lag,
        type = "Ljung-Box"
      )

      if(lb$p.value > 0.05){
        list(
          model = model,
          AIC  = model$aic,
          BIC  = model$bic,
          AICc = model$aicc
        )
      } else NULL

    }, error = function(e) NULL)

  })

  results <- Filter(Negate(is.null), results)

  if(length(results) == 0)
    stop("No ARIMA model passed Ljung-Box test.")

  best <- results[[ which.min(sapply(results, `[[`, criterion)) ]]
  final_model <- best$model
  return(final_model)
}
