suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

DATA_ROOT <- file.path(
  "..",
  "Data"
)

DATASET_NAME <- "linedp_dataset"
MODEL_NAME <- "logistic_regression"

WITH_FILE_LEVEL_ROOT <- file.path(
  DATA_ROOT,
  DATASET_NAME,
  "results",
  "ranking_evaluation",
  MODEL_NAME
)

WITHOUT_FILE_LEVEL_ROOT <- file.path(
  DATA_ROOT,
  DATASET_NAME,
  "results",
  paste0(
    "ranking_evaluation_without_file_prediction_",
    "and_file_counterfactual"
  ),
  MODEL_NAME
)

OUTPUT_ROOT <- file.path(
  ".",
  "result_fig",
  "file_level_components_ablation",
  DATASET_NAME
)

AGGREGATION_METHODS <- c(
  "SUM",
  "MAX",
  "TOP3AVG",
  "TOP2AVG"
)

SUMMARY_STATISTIC <- "mean"
# SUMMARY_STATISTIC <- "median"

METHOD_ORDER <- c(
  "with_file_level_components",
  "without_file_level_components"
)

METHOD_LABELS <- c(
  "without_file_level_components" =
    "CLEAR w/o File-Level Components",
  "with_file_level_components" =
    "CLEAR"
)

METHOD_COLORS <- c(
  "without_file_level_components" =
    "#547DA3",
  "with_file_level_components" =
    "#DA936A"
)

METRIC_ORDER <- c(
  "recall_20",
  "effort@20%recall",
  "far",
  "auc",
  "d2h",
  "ifa"
)

METRIC_LABELS <- c(
  "recall_20" =
    "Recall@20% (\u2191)",
  "effort@20%recall" =
    "Effort@20% (\u2193)",
  "far" =
    "FAR (\u2193)",
  "auc" =
    "AUC (\u2191)",
  "d2h" =
    "D2H (\u2193)",
  "ifa" =
    "IFA (\u2193)"
)

METRIC_DIRECTIONS <- c(
  "recall_20" =
    "higher",
  "effort@20%recall" =
    "lower",
  "far" =
    "lower",
  "auc" =
    "higher",
  "d2h" =
    "lower",
  "ifa" =
    "lower"
)

LEFT_Y_LIMITS <- c(
  0,
  0.8
)

IFA_RIGHT_LIMITS <- NULL
IFA_RIGHT_PADDING_RATIO <- 0.10

LEFT_AXIS_ACCURACY <- 0.01
BAR_LABEL_ACCURACY <- 0.00001
IFA_BAR_LABEL_ACCURACY <- 1

BAR_WIDTH <- 0.68
DODGE_WIDTH <- 0.78

OUTPUT_WIDTH <- 16
OUTPUT_HEIGHT <- 8
OUTPUT_DPI <- 300


validate_configuration <- function()
{
  valid_statistics <- c(
    "mean",
    "median"
  )

  if (!(SUMMARY_STATISTIC %in% valid_statistics)) {
    stop(
      paste0(
        "SUMMARY_STATISTIC must be one of: ",
        paste(
          valid_statistics,
          collapse = ", "
        ),
        "."
      )
    )
  }

  if (length(AGGREGATION_METHODS) != 4) {
    stop(
      "Exactly four aggregation methods must be configured."
    )
  }

  if (
    length(METHOD_ORDER) != 2 ||
    METHOD_ORDER[1] !=
    "with_file_level_components" ||
    METHOD_ORDER[2] !=
    "without_file_level_components"
  ) {
    stop(
      paste0(
        "METHOD_ORDER must place CLEAR first ",
        "and the no-file-level variant second."
      )
    )
  }
}


get_input_filename <- function(
    aggregation_method
)
{
  return(
    paste0(
      "CLEAR-",
      aggregation_method,
      ".csv"
    )
  )
}


get_output_figure_path <- function(
    aggregation_method
)
{
  return(
    file.path(
      OUTPUT_ROOT,
      paste0(
        "RQ_file_level_components_ablation_",
        tolower(
          aggregation_method
        ),
        ".pdf"
      )
    )
  )
}


load_metric_file <- function(
    input_path,
    method_name
)
{
  if (!file.exists(input_path)) {
    stop(
      paste0(
        "Input file does not exist: ",
        input_path
      )
    )
  }

  input_data <- read.csv(
    input_path,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  required_columns <- c(
    "release",
    METRIC_ORDER
  )

  missing_columns <- setdiff(
    required_columns,
    names(
      input_data
    )
  )

  if (length(missing_columns) > 0) {
    stop(
      paste0(
        "The input file is missing the following columns: ",
        paste(
          missing_columns,
          collapse = ", "
        ),
        ". File: ",
        input_path
      )
    )
  }

  input_data <- input_data[
    ,
    required_columns,
    drop = FALSE
  ]

  input_data$release <- as.character(
    input_data$release
  )

  if (
    any(
      duplicated(
        input_data$release
      )
    )
  ) {
    stop(
      paste0(
        "Duplicate release rows were found in ",
        input_path,
        "."
      )
    )
  }

  for (metric_name in METRIC_ORDER) {
    input_data[[
      metric_name
    ]] <- suppressWarnings(
      as.numeric(
        input_data[[
          metric_name
        ]]
      )
    )
  }

  input_data$method <- method_name

  return(
    input_data
  )
}


load_paired_data <- function(
    aggregation_method
)
{
  input_filename <- get_input_filename(
    aggregation_method
  )

  with_file_level_path <- file.path(
    WITH_FILE_LEVEL_ROOT,
    input_filename
  )

  without_file_level_path <- file.path(
    WITHOUT_FILE_LEVEL_ROOT,
    input_filename
  )

  with_file_level_data <- load_metric_file(
    with_file_level_path,
    "with_file_level_components"
  )

  without_file_level_data <- load_metric_file(
    without_file_level_path,
    "without_file_level_components"
  )

  common_releases <- intersect(
    with_file_level_data$release,
    without_file_level_data$release
  )

  if (length(common_releases) == 0) {
    stop(
      paste0(
        "No common releases were found for ",
        aggregation_method,
        "."
      )
    )
  }

  missing_with_file_level <- setdiff(
    without_file_level_data$release,
    with_file_level_data$release
  )

  missing_without_file_level <- setdiff(
    with_file_level_data$release,
    without_file_level_data$release
  )

  if (length(missing_with_file_level) > 0) {
    warning(
      paste0(
        "[",
        aggregation_method,
        "] Releases missing from CLEAR results: ",
        paste(
          missing_with_file_level,
          collapse = ", "
        ),
        ". These releases will be excluded."
      )
    )
  }

  if (length(missing_without_file_level) > 0) {
    warning(
      paste0(
        "[",
        aggregation_method,
        "] Releases missing from the no-file-level results: ",
        paste(
          missing_without_file_level,
          collapse = ", "
        ),
        ". These releases will be excluded."
      )
    )
  }

  with_file_level_data <- with_file_level_data[
    with_file_level_data$release %in%
      common_releases,
    ,
    drop = FALSE
  ]

  without_file_level_data <- without_file_level_data[
    without_file_level_data$release %in%
      common_releases,
    ,
    drop = FALSE
  ]

  with_file_level_data <- with_file_level_data[
    match(
      common_releases,
      with_file_level_data$release
    ),
    ,
    drop = FALSE
  ]

  without_file_level_data <- without_file_level_data[
    match(
      common_releases,
      without_file_level_data$release
    ),
    ,
    drop = FALSE
  ]

  combined_data <- rbind(
    without_file_level_data,
    with_file_level_data
  )

  combined_data$method <- factor(
    combined_data$method,
    levels = METHOD_ORDER
  )

  combined_data$aggregation_method <-
    aggregation_method

  return(
    combined_data
  )
}


get_selected_statistic <- function(
    values
)
{
  if (SUMMARY_STATISTIC == "mean") {
    return(
      mean(
        values
      )
    )
  }

  return(
    median(
      values
    )
  )
}


calculate_t_confidence_interval <- function(
    values,
    confidence_level = 0.95
)
{
  finite_values <- values[
    is.finite(
      values
    )
  ]

  sample_count <- length(
    finite_values
  )

  if (sample_count == 0) {
    return(
      c(
        NA_real_,
        NA_real_
      )
    )
  }

  if (sample_count == 1) {
    return(
      c(
        finite_values[1],
        finite_values[1]
      )
    )
  }

  standard_error <- (
    sd(
      finite_values
    ) /
      sqrt(
        sample_count
      )
  )

  critical_value <- qt(
    1 -
      (
        1 -
          confidence_level
      ) /
        2,
    df =
      sample_count -
        1
  )

  mean_value <- mean(
    finite_values
  )

  return(
    c(
      mean_value -
        critical_value *
        standard_error,
      mean_value +
        critical_value *
        standard_error
    )
  )
}


summarize_metrics <- function(
    input_data
)
{
  summary_rows <- list()
  row_index <- 1

  aggregation_method <- unique(
    input_data$aggregation_method
  )

  if (length(aggregation_method) != 1) {
    stop(
      "Expected exactly one aggregation method."
    )
  }

  for (metric_name in METRIC_ORDER) {
    for (method_name in METHOD_ORDER) {
      metric_values <- suppressWarnings(
        as.numeric(
          input_data[
            as.character(
              input_data$method
            ) == method_name,
            metric_name
          ]
        )
      )

      metric_values <- metric_values[
        is.finite(
          metric_values
        )
      ]

      if (length(metric_values) == 0) {
        stop(
          paste0(
            "No finite values were found for metric ",
            metric_name,
            " and method ",
            method_name,
            "."
          )
        )
      }

      confidence_interval <- (
        calculate_t_confidence_interval(
          metric_values
        )
      )

      summary_rows[[
        row_index
      ]] <- data.frame(
        aggregation_method =
          aggregation_method,
        metric =
          metric_name,
        method =
          method_name,
        selected_value =
          get_selected_statistic(
            metric_values
          ),
        mean_value =
          mean(
            metric_values
          ),
        median_value =
          median(
            metric_values
          ),
        standard_deviation =
          if (
            length(
              metric_values
            ) > 1
          ) {
            sd(
              metric_values
            )
          } else {
            0
          },
        mean_ci_lower =
          confidence_interval[1],
        mean_ci_upper =
          confidence_interval[2],
        release_count =
          length(
            metric_values
          ),
        summary_statistic =
          SUMMARY_STATISTIC,
        stringsAsFactors = FALSE
      )

      row_index <- row_index + 1
    }
  }

  summary_data <- do.call(
    rbind,
    summary_rows
  )

  summary_data$metric <- factor(
    summary_data$metric,
    levels = METRIC_ORDER
  )

  summary_data$method <- factor(
    summary_data$method,
    levels = METHOD_ORDER
  )

  summary_data <- summary_data[
    order(
      summary_data$metric,
      summary_data$method
    ),
    ,
    drop = FALSE
  ]

  return(
    summary_data
  )
}


calculate_paired_rank_biserial <- function(
    improvement_values
)
{
  nonzero_values <- improvement_values[
    is.finite(
      improvement_values
    ) &
      improvement_values != 0
  ]

  if (length(nonzero_values) == 0) {
    return(0)
  }

  absolute_ranks <- rank(
    abs(
      nonzero_values
    ),
    ties.method = "average"
  )

  positive_rank_sum <- sum(
    absolute_ranks[
      nonzero_values > 0
    ]
  )

  negative_rank_sum <- sum(
    absolute_ranks[
      nonzero_values < 0
    ]
  )

  denominator <- (
    positive_rank_sum +
      negative_rank_sum
  )

  if (denominator == 0) {
    return(0)
  }

  return(
    (
      positive_rank_sum -
        negative_rank_sum
    ) /
      denominator
  )
}


calculate_paired_tests <- function(
    input_data
)
{
  result_rows <- list()
  paired_value_rows <- list()
  result_index <- 1
  paired_value_index <- 1

  aggregation_method <- unique(
    input_data$aggregation_method
  )

  if (length(aggregation_method) != 1) {
    stop(
      "Expected exactly one aggregation method."
    )
  }

  without_file_level_data <- input_data[
    as.character(
      input_data$method
    ) == "without_file_level_components",
    ,
    drop = FALSE
  ]

  with_file_level_data <- input_data[
    as.character(
      input_data$method
    ) == "with_file_level_components",
    ,
    drop = FALSE
  ]

  for (metric_name in METRIC_ORDER) {
    paired_data <- merge(
      without_file_level_data[
        ,
        c(
          "release",
          metric_name
        ),
        drop = FALSE
      ],
      with_file_level_data[
        ,
        c(
          "release",
          metric_name
        ),
        drop = FALSE
      ],
      by = "release",
      suffixes = c(
        "_without_file_level",
        "_with_file_level"
      ),
      all = FALSE
    )

    without_values <- suppressWarnings(
      as.numeric(
        paired_data[[
          paste0(
            metric_name,
            "_without_file_level"
          )
        ]]
      )
    )

    with_values <- suppressWarnings(
      as.numeric(
        paired_data[[
          paste0(
            metric_name,
            "_with_file_level"
          )
        ]]
      )
    )

    valid_mask <- (
      is.finite(
        without_values
      ) &
        is.finite(
          with_values
        )
    )

    paired_data <- paired_data[
      valid_mask,
      ,
      drop = FALSE
    ]

    without_values <- without_values[
      valid_mask
    ]

    with_values <- with_values[
      valid_mask
    ]

    if (length(with_values) == 0) {
      stop(
        paste0(
          "No finite paired values were found for metric ",
          metric_name,
          "."
        )
      )
    }

    preferred_direction <- METRIC_DIRECTIONS[[
      metric_name
    ]]

    raw_difference <- (
      with_values -
        without_values
    )

    direction_adjusted_improvement <- if (
      preferred_direction == "higher"
    ) {
      raw_difference
    } else {
      -raw_difference
    }

    improvement_confidence_interval <- (
      calculate_t_confidence_interval(
        direction_adjusted_improvement
      )
    )

    wilcoxon_result <- suppressWarnings(
      wilcox.test(
        with_values,
        without_values,
        paired = TRUE,
        alternative = "two.sided",
        exact = FALSE,
        conf.int = FALSE
      )
    )

    relative_improvement <- rep(
      NA_real_,
      length(
        direction_adjusted_improvement
      )
    )

    valid_baseline_mask <- (
      without_values != 0
    )

    relative_improvement[
      valid_baseline_mask
    ] <- (
      direction_adjusted_improvement[
        valid_baseline_mask
      ] /
        abs(
          without_values[
            valid_baseline_mask
          ]
        ) *
        100
    )

    result_rows[[
      result_index
    ]] <- data.frame(
      aggregation_method =
        aggregation_method,
      metric =
        metric_name,
      metric_label =
        METRIC_LABELS[[
          metric_name
        ]],
      preferred_direction =
        preferred_direction,
      release_count =
        length(
          with_values
        ),
      without_file_level_mean =
        mean(
          without_values
        ),
      with_file_level_mean =
        mean(
          with_values
        ),
      without_file_level_median =
        median(
          without_values
        ),
      with_file_level_median =
        median(
          with_values
        ),
      raw_mean_difference =
        mean(
          raw_difference
        ),
      direction_adjusted_mean_improvement =
        mean(
          direction_adjusted_improvement
        ),
      mean_improvement_ci_lower =
        improvement_confidence_interval[1],
      mean_improvement_ci_upper =
        improvement_confidence_interval[2],
      relative_mean_improvement_percent =
        if (
          mean(
            without_values
          ) == 0
        ) {
          NA_real_
        } else {
          mean(
            direction_adjusted_improvement
          ) /
            abs(
              mean(
                without_values
              )
            ) *
            100
        },
      median_improvement =
        median(
          direction_adjusted_improvement
        ),
      median_relative_improvement_percent =
        if (
          any(
            is.finite(
              relative_improvement
            )
          )
        ) {
          median(
            relative_improvement,
            na.rm = TRUE
          )
        } else {
          NA_real_
        },
      wilcoxon_p_value =
        wilcoxon_result$p.value,
      paired_rank_biserial =
        calculate_paired_rank_biserial(
          direction_adjusted_improvement
        ),
      improved_release_count =
        sum(
          direction_adjusted_improvement > 0
        ),
      tied_release_count =
        sum(
          direction_adjusted_improvement == 0
        ),
      degraded_release_count =
        sum(
          direction_adjusted_improvement < 0
        ),
      stringsAsFactors = FALSE
    )

    paired_value_rows[[
      paired_value_index
    ]] <- data.frame(
      aggregation_method =
        aggregation_method,
      release =
        paired_data$release,
      metric =
        metric_name,
      preferred_direction =
        preferred_direction,
      without_file_level_value =
        without_values,
      with_file_level_value =
        with_values,
      raw_difference =
        raw_difference,
      direction_adjusted_improvement =
        direction_adjusted_improvement,
      relative_improvement_percent =
        relative_improvement,
      stringsAsFactors = FALSE
    )

    result_index <- result_index + 1
    paired_value_index <- paired_value_index + 1
  }

  result_data <- do.call(
    rbind,
    result_rows
  )

  result_data$holm_adjusted_p_value <-
    p.adjust(
      result_data$wilcoxon_p_value,
      method = "holm"
    )

  paired_release_data <- do.call(
    rbind,
    paired_value_rows
  )

  return(
    list(
      tests =
        result_data,
      paired_release_values =
        paired_release_data
    )
  )
}


calculate_ifa_limits <- function(
    summary_data
)
{
  if (!is.null(IFA_RIGHT_LIMITS)) {
    if (
      length(
        IFA_RIGHT_LIMITS
      ) != 2 ||
        any(
          !is.finite(
            IFA_RIGHT_LIMITS
          )
        ) ||
        any(
          IFA_RIGHT_LIMITS <= 0
        ) ||
        IFA_RIGHT_LIMITS[1] >=
          IFA_RIGHT_LIMITS[2]
    ) {
      stop(
        paste0(
          "IFA_RIGHT_LIMITS must contain two increasing ",
          "positive values."
        )
      )
    }

    return(
      IFA_RIGHT_LIMITS
    )
  }

  ifa_values <- summary_data$selected_value[
    as.character(
      summary_data$metric
    ) == "ifa"
  ]

  ifa_values <- ifa_values[
    is.finite(
      ifa_values
    ) &
      ifa_values > 0
  ]

  if (length(ifa_values) == 0) {
    stop(
      paste0(
        "No finite positive IFA ",
        SUMMARY_STATISTIC,
        " values were found."
      )
    )
  }

  minimum_exponent <- floor(
    log10(
      min(
        ifa_values
      )
    )
  )

  maximum_exponent <- ceiling(
    log10(
      max(
        ifa_values
      ) *
        (
          1 +
            IFA_RIGHT_PADDING_RATIO
        )
    )
  )

  if (minimum_exponent >= maximum_exponent) {
    maximum_exponent <- (
      minimum_exponent +
        1
    )
  }

  return(
    c(
      10 ^ minimum_exponent,
      10 ^ maximum_exponent
    )
  )
}


transform_ifa_to_left_axis <- function(
    values,
    ifa_limits
)
{
  if (
    any(
      !is.finite(
        values
      )
    ) ||
      any(
        values <= 0
      )
  ) {
    stop(
      paste0(
        "IFA ",
        SUMMARY_STATISTIC,
        " values must be finite and positive."
      )
    )
  }

  log_ifa_limits <- log10(
    ifa_limits
  )

  ifa_range <- diff(
    log_ifa_limits
  )

  left_range <- diff(
    LEFT_Y_LIMITS
  )

  if (
    !is.finite(
      ifa_range
    ) ||
      ifa_range <= 0
  ) {
    stop(
      paste0(
        "IFA right-axis limits must define a positive ",
        "log-scale range."
      )
    )
  }

  transformed_values <- (
    log10(
      values
    ) -
      log_ifa_limits[1]
  ) /
    ifa_range

  return(
    transformed_values *
      left_range +
      LEFT_Y_LIMITS[1]
  )
}


prepare_plot_data <- function(
    summary_data
)
{
  ifa_limits <- calculate_ifa_limits(
    summary_data
  )

  plot_data <- summary_data
  plot_data$plot_value <- (
    plot_data$selected_value
  )

  ifa_mask <- (
    as.character(
      plot_data$metric
    ) == "ifa"
  )

  plot_data$plot_value[
    ifa_mask
  ] <- transform_ifa_to_left_axis(
    values =
      plot_data$selected_value[
        ifa_mask
      ],
    ifa_limits =
      ifa_limits
  )

  plot_data$value_label <- label_number(
    accuracy =
      BAR_LABEL_ACCURACY,
    big.mark = ","
  )(
    plot_data$selected_value
  )

  plot_data$value_label[
    ifa_mask
  ] <- label_number(
    accuracy =
      IFA_BAR_LABEL_ACCURACY,
    big.mark = ","
  )(
    plot_data$selected_value[
      ifa_mask
    ]
  )

  return(
    list(
      plot_data =
        plot_data,
      ifa_limits =
        ifa_limits
    )
  )
}


build_combined_barplot <- function(
    plot_data,
    ifa_limits,
    aggregation_method
)
{
  left_range <- diff(
    LEFT_Y_LIMITS
  )

  log_ifa_limits <- log10(
    ifa_limits
  )

  ifa_range <- diff(
    log_ifa_limits
  )

  ifa_group_index <- match(
    "ifa",
    METRIC_ORDER
  )

  return(
    ggplot(
      plot_data,
      aes(
        x =
          metric,
        y =
          plot_value,
        fill =
          method,
        group =
          method
      )
    ) +
      annotate(
        "rect",
        xmin =
          ifa_group_index -
            0.5,
        xmax =
          ifa_group_index +
            0.5,
        ymin = -Inf,
        ymax = Inf,
        fill = "#F4F4F4",
        alpha = 0.75
      ) +
      geom_vline(
        xintercept = c(
          ifa_group_index -
            0.5,
          ifa_group_index +
            0.5
        ),
        color = "#C8C8C8",
        linewidth = 0.45,
        linetype = "dashed"
      ) +
      geom_col(
        position = position_dodge(
          width =
            DODGE_WIDTH
        ),
        width =
          BAR_WIDTH,
        color =
          "#3F3F3F",
        linewidth = 0.35,
        na.rm = TRUE
      ) +
      geom_text(
        aes(
          label =
            value_label
        ),
        position = position_dodge(
          width =
            DODGE_WIDTH
        ),
        angle = 90,
        hjust = -0.06,
        vjust = 0.5,
        size = 3.8,
        color = "black",
        na.rm = TRUE
      ) +
      scale_fill_manual(
        values =
          METHOD_COLORS,
        breaks =
          METHOD_ORDER,
        labels =
          METHOD_LABELS[
            METHOD_ORDER
          ],
        drop = FALSE
      ) +
      scale_x_discrete(
        limits =
          METRIC_ORDER,
        labels =
          METRIC_LABELS[
            METRIC_ORDER
          ],
        drop = FALSE
      ) +
      scale_y_continuous(
        breaks = seq(
          LEFT_Y_LIMITS[1],
          LEFT_Y_LIMITS[2],
          by = 0.20
        ),
        labels = label_number(
          accuracy =
            LEFT_AXIS_ACCURACY
        ),
        sec.axis = sec_axis(
          trans = ~ 10 ^ (
            (
              (
                . -
                  LEFT_Y_LIMITS[1]
              ) /
                left_range
            ) *
              ifa_range +
              log_ifa_limits[1]
          ),
          name = "IFA",
          breaks = 10 ^ seq(
            log_ifa_limits[1],
            log_ifa_limits[2],
            by = 1
          ),
          labels = trans_format(
            "log10",
            math_format(
              10 ^ .x
            )
          )
        ),
        expand = expansion(
          mult = c(
            0,
            0.03
          )
        )
      ) +
      coord_cartesian(
        ylim =
          LEFT_Y_LIMITS,
        clip = "off"
      ) +
      guides(
        fill = guide_legend(
          title = NULL,
          nrow = 1,
          byrow = TRUE
        )
      ) +
      labs(
        x = NULL,
        y = "Metric Value",
        title = paste0(
          "Effect of File-Level Components (CLEAR-",
          aggregation_method,
          ")"
        )
      ) +
      theme_minimal() +
      theme(
        text = element_text(
          family = "Arial"
        ),
        plot.background = element_rect(
          fill = "white",
          color = NA
        ),
        panel.background = element_rect(
          fill = "white",
          color = NA
        ),
        panel.grid.major.x =
          element_blank(),
        panel.grid.minor.x =
          element_blank(),
        panel.grid.major.y =
          element_line(
            color = "#D7D7D7",
            linewidth = 0.50
          ),
        panel.grid.minor.y =
          element_blank(),
        plot.title =
          element_text(
            size = 21,
            face = "bold",
            color = "black",
            hjust = 0.5,
            margin = margin(
              b = 12
            )
          ),
        axis.title.x =
          element_blank(),
        axis.title.y.left =
          element_text(
            size = 22,
            color = "black",
            margin = margin(
              r = 12
            )
          ),
        axis.title.y.right =
          element_text(
            size = 22,
            color = "black",
            margin = margin(
              l = 12
            )
          ),
        axis.text.x =
          element_text(
            size = 15,
            color = "black",
            margin = margin(
              t = 10
            )
          ),
        axis.text.y.left =
          element_text(
            size = 18,
            color = "black",
            margin = margin(
              r = 8
            )
          ),
        axis.text.y.right =
          element_text(
            size = 18,
            color = "black",
            margin = margin(
              l = 8
            )
          ),
        legend.position =
          "bottom",
        legend.box.spacing =
          grid::unit(
            0.25,
            "cm"
          ),
        legend.margin = margin(
          t = 4,
          r = 0,
          b = 0,
          l = 0
        ),
        legend.text =
          element_text(
            size = 15
          ),
        legend.key.width =
          grid::unit(
            1.20,
            "cm"
          ),
        legend.key.height =
          grid::unit(
            0.55,
            "cm"
          ),
        legend.spacing.x =
          grid::unit(
            0.30,
            "cm"
          ),
        plot.margin = margin(
          t = 24,
          r = 26,
          b = 16,
          l = 22
        )
      )
  )
}


generate_results_for_aggregation <- function(
    aggregation_method
)
{
  paired_input_data <- load_paired_data(
    aggregation_method
  )

  summary_data <- summarize_metrics(
    paired_input_data
  )

  paired_analysis <- calculate_paired_tests(
    paired_input_data
  )

  prepared_plot_data <- prepare_plot_data(
    summary_data
  )

  plot_object <- build_combined_barplot(
    plot_data =
      prepared_plot_data$plot_data,
    ifa_limits =
      prepared_plot_data$ifa_limits,
    aggregation_method =
      aggregation_method
  )

  output_path <- get_output_figure_path(
    aggregation_method
  )

  ggsave(
    filename =
      output_path,
    plot =
      plot_object,
    width =
      OUTPUT_WIDTH,
    height =
      OUTPUT_HEIGHT,
    units = "in",
    dpi =
      OUTPUT_DPI,
    device =
      grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )

  message(
    paste0(
      "[",
      aggregation_method,
      "] Saved file-level ablation figure to ",
      output_path
    )
  )

  return(
    list(
      summary =
        prepared_plot_data$plot_data,
      tests =
        paired_analysis$tests,
      paired_release_values =
        paired_analysis$paired_release_values
    )
  )
}


validate_configuration()

dir.create(
  OUTPUT_ROOT,
  recursive = TRUE,
  showWarnings = FALSE
)

all_summary_rows <- list()
all_test_rows <- list()
all_paired_release_rows <- list()

for (
  aggregation_index in
  seq_along(
    AGGREGATION_METHODS
  )
) {
  aggregation_method <- AGGREGATION_METHODS[
    aggregation_index
  ]

  aggregation_results <- (
    generate_results_for_aggregation(
      aggregation_method
    )
  )

  all_summary_rows[[
    aggregation_index
  ]] <- aggregation_results$summary

  all_test_rows[[
    aggregation_index
  ]] <- aggregation_results$tests

  all_paired_release_rows[[
    aggregation_index
  ]] <- (
    aggregation_results$paired_release_values
  )
}

all_summary_data <- do.call(
  rbind,
  all_summary_rows
)

all_test_data <- do.call(
  rbind,
  all_test_rows
)

all_test_data$holm_adjusted_p_value_all_tests <-
  p.adjust(
    all_test_data$wilcoxon_p_value,
    method = "holm"
  )

all_paired_release_data <- do.call(
  rbind,
  all_paired_release_rows
)

summary_output_path <- file.path(
  OUTPUT_ROOT,
  paste0(
    "file_level_components_ablation_",
    SUMMARY_STATISTIC,
    "_summary.csv"
  )
)

test_output_path <- file.path(
  OUTPUT_ROOT,
  "file_level_components_ablation_paired_tests.csv"
)

paired_release_output_path <- file.path(
  OUTPUT_ROOT,
  "file_level_components_ablation_paired_release_values.csv"
)

write.csv(
  all_summary_data,
  summary_output_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  all_test_data,
  test_output_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  all_paired_release_data,
  paired_release_output_path,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

message(
  paste0(
    "Saved all four file-level ablation figures and statistics to ",
    OUTPUT_ROOT
  )
)
