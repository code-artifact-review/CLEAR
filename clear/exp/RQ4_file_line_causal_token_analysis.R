suppressPackageStartupMessages({
  library(ggplot2)
  library(scales)
})

SCRIPT_ARGS <- commandArgs(trailingOnly = FALSE)
SCRIPT_FILE_ARG <- grep(
  "^--file=",
  SCRIPT_ARGS,
  value = TRUE
)

if (length(SCRIPT_FILE_ARG) == 0) {
  stop("Unable to determine the R script path.")
}

SCRIPT_FILE <- sub(
  "^--file=",
  "",
  SCRIPT_FILE_ARG[1]
)

SCRIPT_DIR <- dirname(
  normalizePath(
    SCRIPT_FILE,
    winslash = "/",
    mustWork = TRUE
  )
)

PROJECT_ROOT <- normalizePath(
  file.path(
    SCRIPT_DIR,
    "..",
    ".."
  ),
  winslash = "/",
  mustWork = TRUE
)


CLEAR_AGGREGATION_METHODS <- c(
  "sum",
  "max",
  "top3avg",
  "top2avg"
)


run_line_level_causal_token_effectiveness <- function(
    aggregation_method
)
{
  DATASET_NAME <- "linedp_dataset"
  MODEL_NAME <- "logistic_regression"
  aggregation_label <- toupper(
    aggregation_method
  )
  RESULT_FILENAME <- paste0(
    "CLEAR-",
    aggregation_label,
    ".csv"
  )
  
  RESULTS_ROOT <- file.path(
    PROJECT_ROOT,
    "clear",
    "Data",
    DATASET_NAME,
    "results"
  )
  
  OUTPUT_ROOT <- file.path(
    PROJECT_ROOT,
    "clear",
    "exp",
    "result_fig",
    "RQ_causal_token_effectiveness",
    DATASET_NAME,
    "line_level_causal_token",
    aggregation_method
  )
  
  FULL_CLEAR_FILE <- file.path(
    RESULTS_ROOT,
    "ranking_evaluation",
    MODEL_NAME,
    RESULT_FILENAME
  )
  
  WITHOUT_LINE_CAUSAL_TOKEN_FILE <- file.path(
    RESULTS_ROOT,
    "ranking_evaluation_without_line_counterfactual_token",
    MODEL_NAME,
    RESULT_FILENAME
  )
  
  FULL_CLEAR_DIRECTORY_CANDIDATES <- c(
    "ranking_evaluation"
  )
  
  WITHOUT_LINE_DIRECTORY_CANDIDATES <- c(
    "ranking_evaluation_without_line_counterfactual_token",
    "ranking_evaluation_without_line_counterfactual_tokens",
    "ranking_evaluation_without_line_causal_token",
    "ranking_evaluation_without_line_causal_tokens",
    "ranking_evaluation_without_line_token",
    "ranking_evaluation_without_line_tokens"
  )
  
  EFFECT_ORDER <- c(
    "line_level"
  )
  
  LINE_EFFECT_ORDER <- c(
    "line_level"
  )
  
  EFFECT_LABELS <- c(
    "line_level" =
      ""
  )
  
  METRIC_LEVEL_LABELS <- c(
    "line" =
      "Line-Level IFA"
  )
  
  GROUP_ORDER <- c(
    "Effective",
    "Non-Effective"
  )
  
  GROUP_COLORS <- c(
    "Effective" =
      "#547DA3",
    "Non-Effective" =
      "#DA936A"
  )
  
  EFFECT_COLORS <- c(
    "line_level" =
      "#547DA3"
  )
  
  QUARTILE_PROPORTION <- 0.25
  SIGNIFICANCE_LEVEL <- 0.05
  OUTPUT_DPI <- 300
  
  LINE_OVERALL_FIGURE_WIDTH <- 4
  LINE_OVERALL_FIGURE_HEIGHT <- 7
  
  LINE_QUARTILE_FIGURE_WIDTH <- 4
  LINE_QUARTILE_FIGURE_HEIGHT <- 7
  
  LINE_IFA_GAIN_PSEUDO_LOG_SIGMA <- 50
  
  LINE_IFA_GAIN_AXIS_BREAKS <- c(
    -1000,
    -500,
    -200,
    -100,
    -50,
    0,
    50,
    100,
    200,
    500,
    1000,
    2000,
    4000
  )
  
  PER_RELEASE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "line_level_causal_token_effectiveness_per_release.csv"
  )
  
  OVERALL_TEST_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "line_level_causal_token_effectiveness_overall_tests.csv"
  )
  
  PROJECT_SENSITIVITY_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "line_level_causal_token_effectiveness_project_sensitivity.csv"
  )
  
  QUARTILE_ASSIGNMENT_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "line_level_causal_token_effectiveness_quartile_assignments.csv"
  )
  
  QUARTILE_SUMMARY_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "line_level_causal_token_effectiveness_quartile_summary.csv"
  )
  
  QUARTILE_TEST_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "line_level_causal_token_effectiveness_quartile_tests.csv"
  )
  
  LINE_OVERALL_FIGURE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "line_level_causal_token_effectiveness_IFA_overall.pdf"
  )
  
  LINE_QUARTILE_FIGURE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "line_level_causal_token_effectiveness_IFA_effective_non_effective.pdf"
  )
  
  resolve_result_file <- function(
    exact_path,
    directory_candidates,
    configuration_name
  )
  {
    if (!is.null(exact_path)) {
      if (!file.exists(exact_path)) {
        stop(
          paste0(
            "[",
            configuration_name,
            "] The configured result file does not exist: ",
            exact_path
          )
        )
      }
      
      return(
        normalizePath(
          exact_path,
          winslash = "/",
          mustWork = TRUE
        )
      )
    }
    
    candidate_paths <- file.path(
      RESULTS_ROOT,
      directory_candidates,
      MODEL_NAME,
      RESULT_FILENAME
    )
    
    existing_paths <- candidate_paths[
      file.exists(candidate_paths)
    ]
    
    if (length(existing_paths) == 1) {
      return(
        normalizePath(
          existing_paths[1],
          winslash = "/",
          mustWork = TRUE
        )
      )
    }
    
    if (length(existing_paths) > 1) {
      stop(
        paste0(
          "[",
          configuration_name,
          "] Multiple candidate files were found. ",
          "Set the exact file path at the top of the script: ",
          paste(
            existing_paths,
            collapse = "; "
          )
        )
      )
    }
    
    discovered_files <- list.files(
      RESULTS_ROOT,
      pattern = paste0(
        "^",
        gsub(
          "\\.",
          "\\\\.",
          RESULT_FILENAME
        ),
        "$"
      ),
      recursive = TRUE,
      full.names = TRUE,
      ignore.case = TRUE
    )
    
    stop(
      paste0(
        "[",
        configuration_name,
        "] No result file was found in the configured candidate directories.",
        "\nChecked paths:\n",
        paste(
          candidate_paths,
          collapse = "\n"
        ),
        if (length(discovered_files) > 0) {
          paste0(
            "\nAvailable ",
            RESULT_FILENAME,
            " files under RESULTS_ROOT:\n",
            paste(
              discovered_files,
              collapse = "\n"
            )
          )
        } else {
          ""
        }
      )
    )
  }
  
  
  load_line_ifa_results <- function(
    input_path,
    output_name
  )
  {
    input_data <- read.csv(
      input_path,
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    
    normalized_column_names <- tolower(
      gsub(
        "[^A-Za-z0-9]+",
        "",
        names(input_data)
      )
    )
    
    release_index <- match(
      "release",
      normalized_column_names
    )
    
    ifa_index <- match(
      "ifa",
      normalized_column_names
    )
    
    if (
      is.na(release_index) ||
      is.na(ifa_index)
    ) {
      stop(
        paste0(
          "The input file must contain release and IFA columns: ",
          input_path,
          ". Available columns: ",
          paste(
            names(input_data),
            collapse = ", "
          )
        )
      )
    }
    
    output_data <- data.frame(
      release =
        as.character(
          input_data[[release_index]]
        ),
      value =
        suppressWarnings(
          as.numeric(
            input_data[[ifa_index]]
          )
        ),
      stringsAsFactors = FALSE
    )
    
    names(output_data)[2] <- output_name
    
    output_data <- output_data[
      !is.na(output_data$release) &
        nzchar(output_data$release) &
        is.finite(output_data[[output_name]]),
      ,
      drop = FALSE
    ]
    
    if (any(duplicated(output_data$release))) {
      stop(
        paste0(
          "Duplicate release rows were found in ",
          input_path,
          "."
        )
      )
    }
    
    return(output_data)
  }
  
  
  calculate_t_confidence_interval <- function(
    values,
    confidence_level = 0.95
  )
  {
    finite_values <- values[
      is.finite(values)
    ]
    
    sample_count <- length(finite_values)
    
    if (sample_count == 0) {
      return(c(NA_real_, NA_real_))
    }
    
    if (sample_count == 1) {
      return(c(finite_values[1], finite_values[1]))
    }
    
    standard_error <- (
      sd(finite_values) /
        sqrt(sample_count)
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
    
    mean_value <- mean(finite_values)
    
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
  
  
  calculate_signed_rank_biserial <- function(
    values
  )
  {
    nonzero_values <- values[
      is.finite(values) &
        values != 0
    ]
    
    if (length(nonzero_values) == 0) {
      return(0)
    }
    
    absolute_ranks <- rank(
      abs(nonzero_values),
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
  
  
  calculate_cliffs_delta <- function(
    first_values,
    second_values
  )
  {
    first_values <- first_values[
      is.finite(first_values)
    ]
    
    second_values <- second_values[
      is.finite(second_values)
    ]
    
    if (
      length(first_values) == 0 ||
      length(second_values) == 0
    ) {
      return(NA_real_)
    }
    
    comparison_values <- outer(
      first_values,
      second_values,
      "-"
    )
    
    return(
      (
        sum(comparison_values > 0) -
          sum(comparison_values < 0)
      ) /
        length(comparison_values)
    )
  }
  
  
  calculate_overall_tests <- function(
    effect_data
  )
  {
    result_rows <- list()
    
    for (effect_index in seq_along(EFFECT_ORDER)) {
      effect_name <- EFFECT_ORDER[effect_index]
      
      effect_subset <- effect_data[
        as.character(effect_data$effect) ==
          effect_name,
        ,
        drop = FALSE
      ]
      
      effect_values <- effect_subset$absolute_gain[
        is.finite(effect_subset$absolute_gain)
      ]
      
      confidence_interval <- (
        calculate_t_confidence_interval(
          effect_values
        )
      )
      
      wilcoxon_result <- suppressWarnings(
        wilcox.test(
          effect_values,
          mu = 0,
          alternative = "greater",
          exact = FALSE,
          conf.int = FALSE
        )
      )
      
      result_rows[[effect_index]] <- data.frame(
        effect =
          effect_name,
        effect_label =
          EFFECT_LABELS[[effect_name]],
        metric_level =
          as.character(
            effect_subset$metric_level[1]
          ),
        metric_label =
          METRIC_LEVEL_LABELS[[
            as.character(
              effect_subset$metric_level[1]
            )
          ]],
        release_count =
          length(effect_values),
        mean_gain =
          mean(effect_values),
        median_gain =
          median(effect_values),
        standard_deviation =
          if (length(effect_values) > 1) {
            sd(effect_values)
          } else {
            0
          },
        ci_lower =
          confidence_interval[1],
        ci_upper =
          confidence_interval[2],
        wilcoxon_greater_p_value =
          wilcoxon_result$p.value,
        signed_rank_biserial =
          calculate_signed_rank_biserial(
            effect_values
          ),
        improved_release_count =
          sum(effect_values > 0),
        tied_release_count =
          sum(effect_values == 0),
        degraded_release_count =
          sum(effect_values < 0),
        stringsAsFactors = FALSE
      )
    }
    
    result_data <- do.call(
      rbind,
      result_rows
    )
    
    result_data$holm_adjusted_p_value <- p.adjust(
      result_data$wilcoxon_greater_p_value,
      method = "holm"
    )
    
    result_data$effectiveness_supported <- (
      result_data$holm_adjusted_p_value <
        SIGNIFICANCE_LEVEL &
        result_data$median_gain > 0
    )
    
    return(result_data)
  }
  
  
  calculate_project_sensitivity <- function(
    effect_data
  )
  {
    result_rows <- list()
    
    for (effect_index in seq_along(EFFECT_ORDER)) {
      effect_name <- EFFECT_ORDER[effect_index]
      
      effect_subset <- effect_data[
        as.character(effect_data$effect) ==
          effect_name,
        ,
        drop = FALSE
      ]
      
      project_medians <- aggregate(
        absolute_gain ~ project,
        data =
          effect_subset,
        FUN =
          median
      )
      
      project_values <- project_medians$absolute_gain[
        is.finite(
          project_medians$absolute_gain
        )
      ]
      
      wilcoxon_result <- suppressWarnings(
        wilcox.test(
          project_values,
          mu = 0,
          alternative = "greater",
          exact = FALSE,
          conf.int = FALSE
        )
      )
      
      result_rows[[effect_index]] <- data.frame(
        effect =
          effect_name,
        effect_label =
          EFFECT_LABELS[[effect_name]],
        metric_level =
          as.character(
            effect_subset$metric_level[1]
          ),
        project_count =
          length(project_values),
        mean_project_median_gain =
          mean(project_values),
        median_project_median_gain =
          median(project_values),
        wilcoxon_greater_p_value =
          wilcoxon_result$p.value,
        signed_rank_biserial =
          calculate_signed_rank_biserial(
            project_values
          ),
        improved_project_count =
          sum(project_values > 0),
        tied_project_count =
          sum(project_values == 0),
        degraded_project_count =
          sum(project_values < 0),
        stringsAsFactors = FALSE
      )
    }
    
    result_data <- do.call(
      rbind,
      result_rows
    )
    
    result_data$holm_adjusted_p_value <- p.adjust(
      result_data$wilcoxon_greater_p_value,
      method = "holm"
    )
    
    return(result_data)
  }
  
  
  assign_sound_style_groups <- function(
    effect_data
  )
  {
    grouped_rows <- list()
    
    for (effect_index in seq_along(EFFECT_ORDER)) {
      effect_name <- EFFECT_ORDER[effect_index]
      
      effect_subset <- effect_data[
        as.character(effect_data$effect) ==
          effect_name,
        ,
        drop = FALSE
      ]
      
      lower_cutoff <- as.numeric(
        quantile(
          effect_subset$reference_ifa,
          probs =
            QUARTILE_PROPORTION,
          type = 7,
          na.rm = TRUE
        )
      )
      
      upper_cutoff <- as.numeric(
        quantile(
          effect_subset$reference_ifa,
          probs =
            1 -
            QUARTILE_PROPORTION,
          type = 7,
          na.rm = TRUE
        )
      )
      
      if (lower_cutoff >= upper_cutoff) {
        stop(
          paste0(
            "The lower and upper IFA quartile cutoffs overlap for ",
            effect_name,
            "."
          )
        )
      }
      
      effect_subset$group <- NA_character_
      
      effect_subset$group[
        effect_subset$reference_ifa <=
          lower_cutoff
      ] <- "Effective"
      
      effect_subset$group[
        effect_subset$reference_ifa >=
          upper_cutoff
      ] <- "Non-Effective"
      
      effect_subset$lower_ifa_cutoff <-
        lower_cutoff
      
      effect_subset$upper_ifa_cutoff <-
        upper_cutoff
      
      grouped_rows[[effect_index]] <-
        effect_subset
    }
    
    grouped_data <- do.call(
      rbind,
      grouped_rows
    )
    
    grouped_data$group <- factor(
      grouped_data$group,
      levels =
        GROUP_ORDER
    )
    
    return(grouped_data)
  }
  
  
  calculate_quartile_summary <- function(
    grouped_data
  )
  {
    result_rows <- list()
    row_index <- 1
    
    for (effect_name in EFFECT_ORDER) {
      for (group_name in GROUP_ORDER) {
        effect_subset <- grouped_data[
          as.character(grouped_data$effect) ==
            effect_name &
            as.character(grouped_data$group) ==
            group_name,
          ,
          drop = FALSE
        ]
        
        group_values <- effect_subset$absolute_gain[
          is.finite(
            effect_subset$absolute_gain
          )
        ]
        
        confidence_interval <- (
          calculate_t_confidence_interval(
            group_values
          )
        )
        
        result_rows[[row_index]] <- data.frame(
          effect =
            effect_name,
          effect_label =
            EFFECT_LABELS[[effect_name]],
          metric_level =
            as.character(
              effect_subset$metric_level[1]
            ),
          group =
            group_name,
          release_count =
            length(group_values),
          mean_absolute_gain =
            mean(group_values),
          median_absolute_gain =
            median(group_values),
          standard_deviation =
            if (length(group_values) > 1) {
              sd(group_values)
            } else {
              0
            },
          ci_lower =
            confidence_interval[1],
          ci_upper =
            confidence_interval[2],
          improved_release_count =
            sum(group_values > 0),
          tied_release_count =
            sum(group_values == 0),
          degraded_release_count =
            sum(group_values < 0),
          stringsAsFactors = FALSE
        )
        
        row_index <- row_index + 1
      }
    }
    
    return(
      do.call(
        rbind,
        result_rows
      )
    )
  }
  
  
  calculate_quartile_tests <- function(
    grouped_data
  )
  {
    within_rows <- list()
    between_rows <- list()
    within_index <- 1
    between_index <- 1
    
    for (effect_name in EFFECT_ORDER) {
      effect_subset <- grouped_data[
        as.character(grouped_data$effect) ==
          effect_name,
        ,
        drop = FALSE
      ]
      
      for (group_name in GROUP_ORDER) {
        group_values <- effect_subset$absolute_gain[
          as.character(effect_subset$group) ==
            group_name
        ]
        
        group_values <- group_values[
          is.finite(group_values)
        ]
        
        wilcoxon_result <- suppressWarnings(
          wilcox.test(
            group_values,
            mu = 0,
            alternative = "greater",
            exact = FALSE,
            conf.int = FALSE
          )
        )
        
        within_rows[[within_index]] <- data.frame(
          analysis_type =
            "within_group_gain_greater_than_zero",
          effect =
            effect_name,
          metric_level =
            as.character(
              effect_subset$metric_level[1]
            ),
          group =
            group_name,
          first_group =
            group_name,
          second_group =
            NA_character_,
          sample_size_first =
            length(group_values),
          sample_size_second =
            NA_integer_,
          p_value =
            wilcoxon_result$p.value,
          effect_size =
            calculate_signed_rank_biserial(
              group_values
            ),
          effect_size_name =
            "signed_rank_biserial",
          stringsAsFactors = FALSE
        )
        
        within_index <- within_index + 1
      }
      
      effective_values <- effect_subset$absolute_gain[
        as.character(effect_subset$group) ==
          "Effective"
      ]
      
      non_effective_values <- effect_subset$absolute_gain[
        as.character(effect_subset$group) ==
          "Non-Effective"
      ]
      
      group_test <- suppressWarnings(
        wilcox.test(
          non_effective_values,
          effective_values,
          alternative = "two.sided",
          paired = FALSE,
          exact = FALSE,
          conf.int = FALSE
        )
      )
      
      between_rows[[between_index]] <- data.frame(
        analysis_type =
          "between_group_gain_difference",
        effect =
          effect_name,
        metric_level =
          as.character(
            effect_subset$metric_level[1]
          ),
        group =
          NA_character_,
        first_group =
          "Non-Effective",
        second_group =
          "Effective",
        sample_size_first =
          sum(is.finite(non_effective_values)),
        sample_size_second =
          sum(is.finite(effective_values)),
        p_value =
          group_test$p.value,
        effect_size =
          calculate_cliffs_delta(
            non_effective_values,
            effective_values
          ),
        effect_size_name =
          "cliffs_delta_non_effective_minus_effective",
        stringsAsFactors = FALSE
      )
      
      between_index <- between_index + 1
    }
    
    within_data <- do.call(
      rbind,
      within_rows
    )
    
    between_data <- do.call(
      rbind,
      between_rows
    )
    
    within_data$holm_adjusted_p_value <- p.adjust(
      within_data$p_value,
      method = "holm"
    )
    
    between_data$holm_adjusted_p_value <- p.adjust(
      between_data$p_value,
      method = "holm"
    )
    
    return(
      rbind(
        within_data,
        between_data
      )
    )
  }
  
  
  build_overall_plot <- function(
    effect_data,
    overall_results,
    effect_order,
    title,
    subtitle,
    y_axis_title,
    use_pseudo_log
  )
  {
    plot_data <- effect_data[
      as.character(effect_data$effect) %in%
        effect_order,
      ,
      drop = FALSE
    ]
    
    plot_data$effect <- factor(
      as.character(plot_data$effect),
      levels =
        effect_order
    )
    
    plot_data$effect_label <- factor(
      unname(
        EFFECT_LABELS[
          as.character(plot_data$effect)
        ]
      ),
      levels =
        unname(
          EFFECT_LABELS[
            effect_order
          ]
        )
    )
    
    summary_data <- overall_results[
      overall_results$effect %in%
        effect_order,
      ,
      drop = FALSE
    ]
    
    summary_data$effect_label <- factor(
      summary_data$effect_label,
      levels =
        unname(
          EFFECT_LABELS[
            effect_order
          ]
        )
    )
    
    y_scale <- if (use_pseudo_log) {
      scale_y_continuous(
        trans = pseudo_log_trans(
          sigma =
            LINE_IFA_GAIN_PSEUDO_LOG_SIGMA,
          base = 10
        ),
        breaks =
          LINE_IFA_GAIN_AXIS_BREAKS,
        labels = label_number(
          accuracy = 1,
          big.mark = ","
        ),
        expand = expansion(
          mult = c(
            0.08,
            0.10
          )
        )
      )
    } else {
      scale_y_continuous(
        breaks = pretty_breaks(
          n = 7
        ),
        labels = label_number(
          accuracy = 1,
          big.mark = ","
        ),
        expand = expansion(
          mult = c(
            0.08,
            0.10
          )
        )
      )
    }
    
    return(
      ggplot(
        plot_data,
        aes(
          x =
            effect_label,
          y =
            absolute_gain,
          color =
            effect
        )
      ) +
        geom_hline(
          yintercept = 0,
          color = "#8F8F8F",
          linewidth = 0.65,
          linetype = "dashed"
        ) +
        geom_boxplot(
          aes(
            group =
              effect_label,
            fill =
              effect
          ),
          width = 0.48,
          alpha = 0.16,
          color = "#606060",
          linewidth = 0.60,
          outlier.shape = NA,
          show.legend = FALSE
        ) +
        geom_errorbar(
          data =
            summary_data,
          aes(
            x =
              effect_label,
            ymin =
              ci_lower,
            ymax =
              ci_upper
          ),
          inherit.aes = FALSE,
          width = 0.10,
          linewidth = 0.85,
          color = "black"
        ) +
        geom_jitter(
          width = 0.085,
          height = 0,
          size = 2.8,
          alpha = 0.78,
          show.legend = FALSE
        ) +
        geom_point(
          data =
            summary_data,
          aes(
            x =
              effect_label,
            y =
              mean_gain
          ),
          inherit.aes = FALSE,
          size = 4.5,
          shape = 21,
          fill = "white",
          color = "black",
          stroke = 1.1
        ) +
        scale_color_manual(
          values =
            EFFECT_COLORS,
          breaks =
            effect_order,
          drop = FALSE
        ) +
        scale_fill_manual(
          values =
            EFFECT_COLORS,
          breaks =
            effect_order,
          drop = FALSE
        ) +
        y_scale +
        labs(
          x = NULL,
          y =
            y_axis_title,
          title =
            title,
          subtitle =
            subtitle
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
          panel.grid.minor =
            element_blank(),
          panel.grid.major.y =
            element_line(
              color = "#D7D7D7",
              linewidth = 0.50
            ),
          plot.title =
            element_text(
              size = 20,
              face = "bold",
              color = "black",
              hjust = 0.5,
              margin = margin(
                b = 7
              )
            ),
          plot.subtitle =
            element_text(
              size = 11.5,
              color = "#2F2F2F",
              hjust = 0.5,
              lineheight = 1.10,
              margin = margin(
                b = 12
              )
            ),
          axis.title.y =
            element_text(
              size = 18,
              color = "black",
              margin = margin(
                r = 12
              )
            ),
          axis.text.x =
            element_text(
              size = 14,
              color = "black",
              margin = margin(
                t = 8
              )
            ),
          axis.text.y =
            element_text(
              size = 14,
              color = "black"
            ),
          legend.position =
            "none",
          plot.margin = margin(
            t = 20,
            r = 24,
            b = 18,
            l = 24
          )
        )
    )
  }
  
  
  build_quartile_plot <- function(
    grouped_data,
    quartile_summary,
    effect_order,
    title,
    subtitle,
    y_axis_title,
    use_pseudo_log
  )
  {
    plot_data <- grouped_data[
      !is.na(grouped_data$group) &
        as.character(grouped_data$effect) %in%
        effect_order,
      ,
      drop = FALSE
    ]
    
    plot_data$effect_panel <- factor(
      unname(
        EFFECT_LABELS[
          as.character(plot_data$effect)
        ]
      ),
      levels =
        unname(
          EFFECT_LABELS[
            effect_order
          ]
        )
    )
    
    plot_data$group <- factor(
      as.character(plot_data$group),
      levels =
        GROUP_ORDER
    )
    
    summary_data <- quartile_summary[
      quartile_summary$effect %in%
        effect_order,
      ,
      drop = FALSE
    ]
    
    summary_data$effect_panel <- factor(
      summary_data$effect_label,
      levels =
        unname(
          EFFECT_LABELS[
            effect_order
          ]
        )
    )
    
    summary_data$group <- factor(
      as.character(summary_data$group),
      levels =
        GROUP_ORDER
    )
    
    y_scale <- if (use_pseudo_log) {
      scale_y_continuous(
        trans = pseudo_log_trans(
          sigma =
            LINE_IFA_GAIN_PSEUDO_LOG_SIGMA,
          base = 10
        ),
        breaks =
          LINE_IFA_GAIN_AXIS_BREAKS,
        labels = label_number(
          accuracy = 1,
          big.mark = ","
        ),
        expand = expansion(
          mult = c(
            0.08,
            0.10
          )
        )
      )
    } else {
      scale_y_continuous(
        breaks = pretty_breaks(
          n = 7
        ),
        labels = label_number(
          accuracy = 1,
          big.mark = ","
        ),
        expand = expansion(
          mult = c(
            0.08,
            0.10
          )
        )
      )
    }
    
    return(
      ggplot(
        plot_data,
        aes(
          x =
            group,
          y =
            absolute_gain,
          color =
            group
        )
      ) +
        geom_hline(
          yintercept = 0,
          color = "#8F8F8F",
          linewidth = 0.65,
          linetype = "dashed"
        ) +
        geom_boxplot(
          aes(
            group =
              group,
            fill =
              group
          ),
          width = 0.52,
          alpha = 0.16,
          color = "#606060",
          linewidth = 0.60,
          outlier.shape = NA,
          show.legend = FALSE
        ) +
        geom_errorbar(
          data =
            summary_data,
          aes(
            x =
              group,
            ymin =
              ci_lower,
            ymax =
              ci_upper
          ),
          inherit.aes = FALSE,
          width = 0.10,
          linewidth = 0.80,
          color = "black"
        ) +
        geom_jitter(
          width = 0.075,
          height = 0,
          size = 2.9,
          alpha = 0.82,
          show.legend = FALSE
        ) +
        geom_point(
          data =
            summary_data,
          aes(
            x =
              group,
            y =
              mean_absolute_gain,
            fill =
              group
          ),
          inherit.aes = FALSE,
          size = 4.3,
          shape = 21,
          color = "black",
          stroke = 1.0
        ) +
        facet_grid(
          cols = vars(
            effect_panel
          ),
          scales = "fixed",
          space = "fixed",
          drop = FALSE
        ) +
        scale_color_manual(
          values =
            GROUP_COLORS,
          breaks =
            GROUP_ORDER,
          drop = FALSE
        ) +
        scale_fill_manual(
          values =
            GROUP_COLORS,
          breaks =
            GROUP_ORDER,
          drop = FALSE
        ) +
        y_scale +
        scale_x_discrete(
          limits =
            GROUP_ORDER,
          drop = FALSE
        ) +
        labs(
          x = NULL,
          y =
            y_axis_title,
          title =
            title,
          subtitle =
            subtitle
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
          panel.grid.minor =
            element_blank(),
          panel.grid.major.y =
            element_line(
              color = "#D7D7D7",
              linewidth = 0.50
            ),
          strip.background =
            element_blank(),
          strip.text =
            element_blank(),
          plot.title =
            element_text(
              size = 20,
              face = "bold",
              color = "black",
              hjust = 0.5,
              margin = margin(
                b = 7
              )
            ),
          plot.subtitle =
            element_text(
              size = 11.2,
              color = "#2F2F2F",
              hjust = 0.5,
              lineheight = 1.10,
              margin = margin(
                b = 12
              )
            ),
          axis.title.y =
            element_text(
              size = 18,
              color = "black",
              margin = margin(
                r = 12
              )
            ),
          axis.text.x =
            element_text(
              size = 12.5,
              color = "black",
              angle = 90,
              hjust = 1,
              vjust = 0.5,
              margin = margin(
                t = 8
              )
            ),
          axis.text.y =
            element_text(
              size = 14,
              color = "black"
            ),
          legend.position =
            "none",
          panel.spacing =
            grid::unit(
              1.0,
              "cm"
            ),
          plot.margin = margin(
            t = 20,
            r = 24,
            b = 34,
            l = 24
          )
        )
    )
  }
  
  prepare_line_effect_data <- function(
    line_merged_data
  )
  {
    effect_data <- data.frame(
      release =
        line_merged_data$release,
      project =
        sub(
          "-.*$",
          "",
          line_merged_data$release
        ),
      effect =
        "line_level",
      metric_level =
        "line",
      full_ifa =
        line_merged_data$ifa_full,
      reference_ifa =
        line_merged_data$ifa_without_line,
      absolute_gain =
        line_merged_data$ifa_without_line -
        line_merged_data$ifa_full,
      stringsAsFactors = FALSE
    )
    
    effect_data$effect <- factor(
      effect_data$effect,
      levels =
        EFFECT_ORDER
    )
    
    effect_data$metric_level <- factor(
      effect_data$metric_level,
      levels =
        "line"
    )
    
    return(effect_data)
  }
  
  dir.create(
    OUTPUT_ROOT,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  full_result_path <- resolve_result_file(
    exact_path =
      FULL_CLEAR_FILE,
    directory_candidates =
      FULL_CLEAR_DIRECTORY_CANDIDATES,
    configuration_name =
      "Full CLEAR"
  )
  
  without_line_result_path <- resolve_result_file(
    exact_path =
      WITHOUT_LINE_CAUSAL_TOKEN_FILE,
    directory_candidates =
      WITHOUT_LINE_DIRECTORY_CANDIDATES,
    configuration_name =
      "Without line-level causal token"
  )
  
  message(
    paste0(
      "Full CLEAR line-level result: ",
      full_result_path
    )
  )
  
  message(
    paste0(
      "Without line-level causal token result: ",
      without_line_result_path
    )
  )
  
  full_line_data <- load_line_ifa_results(
    full_result_path,
    "ifa_full"
  )
  
  without_line_data <- load_line_ifa_results(
    without_line_result_path,
    "ifa_without_line"
  )
  
  line_merged_data <- merge(
    full_line_data,
    without_line_data,
    by = "release",
    all = FALSE
  )
  
  if (nrow(line_merged_data) == 0) {
    stop(
      paste0(
        "No common releases were found between the full CLEAR ",
        "and without-line-token configurations."
      )
    )
  }
  
  effect_data <- prepare_line_effect_data(
    line_merged_data
  )
  
  overall_results <- calculate_overall_tests(
    effect_data
  )
  
  project_sensitivity_results <- (
    calculate_project_sensitivity(
      effect_data
    )
  )
  
  grouped_data <- assign_sound_style_groups(
    effect_data
  )
  
  quartile_summary <- calculate_quartile_summary(
    grouped_data
  )
  
  quartile_tests <- calculate_quartile_tests(
    grouped_data
  )
  
  write.csv(
    effect_data,
    PER_RELEASE_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    overall_results,
    OVERALL_TEST_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    project_sensitivity_results,
    PROJECT_SENSITIVITY_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    grouped_data,
    QUARTILE_ASSIGNMENT_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    quartile_summary,
    QUARTILE_SUMMARY_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    quartile_tests,
    QUARTILE_TEST_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  line_overall_plot <- build_overall_plot(
    effect_data =
      effect_data,
    overall_results =
      overall_results,
    effect_order =
      LINE_EFFECT_ORDER,
    title = NULL,
    subtitle = NULL,
    y_axis_title = expression(
      Delta *
        IFA[line] ==
        IFA[without] -
        IFA[CLEAR]
    ),
    use_pseudo_log =
      TRUE
  )
  
  line_quartile_plot <- build_quartile_plot(
    grouped_data =
      grouped_data,
    quartile_summary =
      quartile_summary,
    effect_order =
      LINE_EFFECT_ORDER,
    title = NULL,
    subtitle = NULL,
    y_axis_title = expression(
      Delta *
        IFA[line] ==
        IFA[without] -
        IFA[CLEAR]
    ),
    use_pseudo_log =
      TRUE
  )
  
  ggsave(
    filename =
      LINE_OVERALL_FIGURE_OUTPUT_PATH,
    plot =
      line_overall_plot,
    width =
      LINE_OVERALL_FIGURE_WIDTH,
    height =
      LINE_OVERALL_FIGURE_HEIGHT,
    units = "in",
    dpi =
      OUTPUT_DPI,
    device =
      grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )
  
  ggsave(
    filename =
      LINE_QUARTILE_FIGURE_OUTPUT_PATH,
    plot =
      line_quartile_plot,
    width =
      LINE_QUARTILE_FIGURE_WIDTH,
    height =
      LINE_QUARTILE_FIGURE_HEIGHT,
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
      "Analyzed ",
      length(
        unique(effect_data$release)
      ),
      " releases for line-level causal-token effectiveness."
    )
  )
  
  message(
    paste0(
      "Saved line-level statistical results and figures to ",
      OUTPUT_ROOT
    )
  )
}



run_file_level_causal_token_discrimination <- function(
    aggregation_method
)
{
  DATASET_NAME <- "linedp_dataset"
  MODEL_NAME <- "logistic_regression"
  AGGREGATION_METHOD <- aggregation_method
  
  INPUT_ROOT <- file.path(
    PROJECT_ROOT,
    "clear",
    "Data",
    DATASET_NAME,
    "results",
    "file_level_causal_token_effectiveness",
    MODEL_NAME,
    AGGREGATION_METHOD
  )
  
  OUTPUT_ROOT <- file.path(
    PROJECT_ROOT,
    "clear",
    "exp",
    "result_fig",
    "RQ_causal_token_effectiveness",
    DATASET_NAME,
    "file_level_causal_token",
    AGGREGATION_METHOD
  )
  
  RANKING_FILE_PATTERN <- "_full_file_ranking\\.csv$"
  SIGNIFICANCE_LEVEL <- 0.05
  OUTPUT_DPI <- 300
  
  DEFECT_RATE_FIGURE_WIDTH <- 4.5
  DEFECT_RATE_FIGURE_HEIGHT <- 7
  
  ODDS_RATIO_FIGURE_WIDTH <- 13
  ODDS_RATIO_FIGURE_HEIGHT <- 9.5
  
  NO_HIT_COLOR <- "#547DA3"
  HIT_COLOR <- "#DA936A"
  NEUTRAL_COLOR <- "#8F8F8F"
  SUMMARY_COLOR <- "#2F2F2F"
  
  PER_RELEASE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "file_level_causal_token_discrimination_per_release.csv"
  )
  
  POOLED_CONTINGENCY_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "file_level_causal_token_pooled_contingency.csv"
  )
  
  POOLED_TEST_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "file_level_causal_token_pooled_tests.csv"
  )
  
  LOGISTIC_MODEL_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "file_level_causal_token_logistic_models.csv"
  )
  
  DEFECT_RATE_FIGURE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "file_level_causal_token_defect_rate_by_release.pdf"
  )
  
  ODDS_RATIO_FIGURE_OUTPUT_PATH <- file.path(
    OUTPUT_ROOT,
    "file_level_causal_token_odds_ratio_forest.pdf"
  )
  
  safe_divide <- function(numerator, denominator) {
    ifelse(
      denominator == 0,
      NA_real_,
      numerator / denominator
    )
  }
  
  
  format_p_value <- function(p_value) {
    if (!is.finite(p_value)) {
      return("p = NA")
    }
    
    if (p_value < 0.001) {
      return("p < 0.001")
    }
    
    paste0(
      "p = ",
      formatC(
        p_value,
        format = "f",
        digits = 3
      )
    )
  }
  
  calculate_t_confidence_interval <- function(
    values,
    confidence_level = 0.95
  ) {
    values <- values[is.finite(values)]
    sample_count <- length(values)
    
    if (sample_count == 0) {
      return(c(NA_real_, NA_real_))
    }
    
    if (sample_count == 1) {
      return(c(values[1], values[1]))
    }
    
    mean_value <- mean(values)
    standard_error <- sd(values) / sqrt(sample_count)
    critical_value <- qt(
      1 - (1 - confidence_level) / 2,
      df = sample_count - 1
    )
    
    c(
      mean_value - critical_value * standard_error,
      mean_value + critical_value * standard_error
    )
  }
  
  load_file_level_ranking <- function(input_path) {
    input_data <- read.csv(
      input_path,
      check.names = FALSE,
      stringsAsFactors = FALSE,
      fileEncoding = "UTF-8"
    )
    
    required_columns <- c(
      "filename",
      "file_label",
      "file_contains_file_counterfactual_token",
      "line_count"
    )
    
    missing_columns <- setdiff(
      required_columns,
      names(input_data)
    )
    
    if (length(missing_columns) > 0) {
      stop(
        paste0(
          "The ranking file is missing required columns: ",
          paste(missing_columns, collapse = ", "),
          ". File: ",
          input_path
        )
      )
    }
    
    release_name <- sub(
      RANKING_FILE_PATTERN,
      "",
      basename(input_path)
    )
    
    output_data <- data.frame(
      test_release = release_name,
      project = sub("-.*$", "", release_name),
      filename = as.character(input_data$filename),
      file_label = suppressWarnings(
        as.integer(input_data$file_label)
      ),
      token_hit = suppressWarnings(
        as.integer(
          input_data$file_contains_file_counterfactual_token
        )
      ),
      line_count = suppressWarnings(
        as.numeric(input_data$line_count)
      ),
      stringsAsFactors = FALSE
    )
    
    valid_mask <- (
      !is.na(output_data$filename) &
        nzchar(output_data$filename) &
        output_data$file_label %in% c(0L, 1L) &
        output_data$token_hit %in% c(0L, 1L) &
        is.finite(output_data$line_count) &
        output_data$line_count >= 0
    )
    
    output_data <- output_data[
      valid_mask,
      ,
      drop = FALSE
    ]
    
    if (any(duplicated(output_data$filename))) {
      duplicate_files <- unique(
        output_data$filename[
          duplicated(output_data$filename)
        ]
      )
      
      stop(
        paste0(
          "Duplicate file rows were found in ",
          input_path,
          ": ",
          paste(head(duplicate_files, 10), collapse = ", ")
        )
      )
    }
    
    if (nrow(output_data) == 0) {
      stop(
        paste0(
          "No valid file rows were found in ",
          input_path,
          "."
        )
      )
    }
    
    output_data
  }
  
  calculate_release_statistics <- function(release_data) {
    a <- sum(
      release_data$token_hit == 1 &
        release_data$file_label == 1
    )
    b <- sum(
      release_data$token_hit == 1 &
        release_data$file_label == 0
    )
    c <- sum(
      release_data$token_hit == 0 &
        release_data$file_label == 1
    )
    d <- sum(
      release_data$token_hit == 0 &
        release_data$file_label == 0
    )
    
    contingency_table <- matrix(
      c(a, b, c, d),
      nrow = 2,
      byrow = TRUE,
      dimnames = list(
        token_hit = c("Yes", "No"),
        defective_file = c("Yes", "No")
      )
    )
    
    fisher_result <- tryCatch(
      fisher.test(
        contingency_table,
        alternative = "two.sided"
      ),
      error = function(error_condition) NULL
    )
    
    fisher_odds_ratio <- NA_real_
    fisher_ci_lower <- NA_real_
    fisher_ci_upper <- NA_real_
    fisher_p_value <- NA_real_
    
    if (!is.null(fisher_result)) {
      if (length(fisher_result$estimate) == 1) {
        fisher_odds_ratio <- unname(fisher_result$estimate)
      }
      if (length(fisher_result$conf.int) == 2) {
        fisher_ci_lower <- fisher_result$conf.int[1]
        fisher_ci_upper <- fisher_result$conf.int[2]
      }
      fisher_p_value <- fisher_result$p.value
    }
    
    corrected_cells <- c(a, b, c, d)
    if (any(corrected_cells == 0)) {
      corrected_cells <- corrected_cells + 0.5
    }
    
    corrected_odds_ratio <- (
      corrected_cells[1] * corrected_cells[4] /
        (corrected_cells[2] * corrected_cells[3])
    )
    
    corrected_standard_error <- sqrt(
      sum(1 / corrected_cells)
    )
    
    corrected_ci_lower <- exp(
      log(corrected_odds_ratio) -
        qnorm(0.975) * corrected_standard_error
    )
    
    corrected_ci_upper <- exp(
      log(corrected_odds_ratio) +
        qnorm(0.975) * corrected_standard_error
    )
    
    hit_file_count <- a + b
    no_hit_file_count <- c + d
    total_defective_files <- a + c
    total_clean_files <- b + d
    
    hit_defect_rate <- safe_divide(a, hit_file_count)
    no_hit_defect_rate <- safe_divide(c, no_hit_file_count)
    
    risk_ratio <- NA_real_
    if (
      is.finite(hit_defect_rate) &&
      is.finite(no_hit_defect_rate) &&
      no_hit_defect_rate > 0
    ) {
      risk_ratio <- hit_defect_rate / no_hit_defect_rate
    }
    
    data.frame(
      test_release = release_data$test_release[1],
      project = release_data$project[1],
      total_files = nrow(release_data),
      total_defective_files = total_defective_files,
      total_clean_files = total_clean_files,
      token_hit_files = hit_file_count,
      token_no_hit_files = no_hit_file_count,
      defective_token_hit = a,
      clean_token_hit = b,
      defective_token_no_hit = c,
      clean_token_no_hit = d,
      token_hit_rate = safe_divide(
        hit_file_count,
        nrow(release_data)
      ),
      token_hit_defect_rate = hit_defect_rate,
      token_no_hit_defect_rate = no_hit_defect_rate,
      defect_rate_difference = (
        hit_defect_rate - no_hit_defect_rate
      ),
      risk_ratio = risk_ratio,
      fisher_odds_ratio = fisher_odds_ratio,
      fisher_ci_lower = fisher_ci_lower,
      fisher_ci_upper = fisher_ci_upper,
      fisher_p_value = fisher_p_value,
      corrected_odds_ratio = corrected_odds_ratio,
      corrected_ci_lower = corrected_ci_lower,
      corrected_ci_upper = corrected_ci_upper,
      sensitivity = safe_divide(a, total_defective_files),
      specificity = safe_divide(d, total_clean_files),
      precision = hit_defect_rate,
      cmh_eligible = (
        hit_file_count > 0 &&
          no_hit_file_count > 0 &&
          total_defective_files > 0 &&
          total_clean_files > 0
      ),
      median_line_count_token_hit = if (hit_file_count > 0) {
        median(
          release_data$line_count[
            release_data$token_hit == 1
          ]
        )
      } else {
        NA_real_
      },
      median_line_count_token_no_hit = if (no_hit_file_count > 0) {
        median(
          release_data$line_count[
            release_data$token_hit == 0
          ]
        )
      } else {
        NA_real_
      },
      stringsAsFactors = FALSE
    )
  }
  
  build_cmh_array <- function(
    all_file_data,
    eligible_releases
  ) {
    cmh_array <- array(
      0,
      dim = c(2, 2, length(eligible_releases)),
      dimnames = list(
        token_hit = c("Yes", "No"),
        defective_file = c("Yes", "No"),
        release = eligible_releases
      )
    )
    
    for (release_index in seq_along(eligible_releases)) {
      release_name <- eligible_releases[release_index]
      release_data <- all_file_data[
        all_file_data$test_release == release_name,
        ,
        drop = FALSE
      ]
      
      cmh_array[, , release_index] <- matrix(
        c(
          sum(
            release_data$token_hit == 1 &
              release_data$file_label == 1
          ),
          sum(
            release_data$token_hit == 1 &
              release_data$file_label == 0
          ),
          sum(
            release_data$token_hit == 0 &
              release_data$file_label == 1
          ),
          sum(
            release_data$token_hit == 0 &
              release_data$file_label == 0
          )
        ),
        nrow = 2,
        byrow = TRUE
      )
    }
    
    cmh_array
  }
  
  extract_logistic_term <- function(
    model,
    term_name,
    analysis_name,
    interpretation
  ) {
    coefficient_table <- summary(model)$coefficients
    
    if (!(term_name %in% rownames(coefficient_table))) {
      return(
        data.frame(
          analysis = analysis_name,
          term = term_name,
          coefficient = NA_real_,
          standard_error = NA_real_,
          odds_ratio = NA_real_,
          ci_lower = NA_real_,
          ci_upper = NA_real_,
          p_value = NA_real_,
          sample_size = nobs(model),
          converged = isTRUE(model$converged),
          interpretation = interpretation,
          stringsAsFactors = FALSE
        )
      )
    }
    
    coefficient_value <- coefficient_table[
      term_name,
      "Estimate"
    ]
    standard_error <- coefficient_table[
      term_name,
      "Std. Error"
    ]
    p_value <- coefficient_table[
      term_name,
      "Pr(>|z|)"
    ]
    
    data.frame(
      analysis = analysis_name,
      term = term_name,
      coefficient = coefficient_value,
      standard_error = standard_error,
      odds_ratio = exp(coefficient_value),
      ci_lower = exp(
        coefficient_value - qnorm(0.975) * standard_error
      ),
      ci_upper = exp(
        coefficient_value + qnorm(0.975) * standard_error
      ),
      p_value = p_value,
      sample_size = nobs(model),
      converged = isTRUE(model$converged),
      interpretation = interpretation,
      stringsAsFactors = FALSE
    )
  }
  
  build_defect_rate_plot <- function(
    release_statistics,
    pooled_tests
  ) {
    difference_data <- release_statistics[
      is.finite(
        release_statistics$token_hit_defect_rate
      ) &
        is.finite(
          release_statistics$token_no_hit_defect_rate
        ),
      ,
      drop = FALSE
    ]
    
    if (nrow(difference_data) == 0) {
      stop(
        "No release has both token-hit and token-no-hit defect rates."
      )
    }
    
    difference_data$defect_rate_difference <- (
      difference_data$token_hit_defect_rate -
        difference_data$token_no_hit_defect_rate
    )
    
    difference_data$direction <- ifelse(
      difference_data$defect_rate_difference > 0,
      "Positive",
      ifelse(
        difference_data$defect_rate_difference < 0,
        "Negative",
        "Tied"
      )
    )
    
    direction_levels <- c(
      "Positive",
      "Tied",
      "Negative"
    )
    
    difference_data$direction <- factor(
      difference_data$direction,
      levels =
        direction_levels
    )
    
    difference_data$comparison <- factor(
      "Token hit minus no hit",
      levels = "Token hit minus no hit"
    )
    
    confidence_interval <- calculate_t_confidence_interval(
      difference_data$defect_rate_difference
    )
    
    summary_data <- data.frame(
      comparison = factor(
        "Token hit minus no hit",
        levels = "Token hit minus no hit"
      ),
      mean_difference = mean(
        difference_data$defect_rate_difference
      ),
      ci_lower = confidence_interval[1],
      ci_upper = confidence_interval[2],
      stringsAsFactors = FALSE
    )
    

    legend_direction_data <- data.frame(
      comparison = factor(
        rep(
          "Token hit minus no hit",
          length(
            direction_levels
          )
        ),
        levels = "Token hit minus no hit"
      ),
      defect_rate_difference = rep(
        0,
        length(
          direction_levels
        )
      ),
      direction = factor(
        direction_levels,
        levels =
          direction_levels
      ),
      stringsAsFactors = FALSE
    )
    
    paired_row <- pooled_tests[
      pooled_tests$analysis ==
        "Paired release-level defect-rate comparison",
      ,
      drop = FALSE
    ]
    
    paired_p_value <- if (nrow(paired_row) == 1) {
      paired_row$p_value[1]
    } else {
      NA_real_
    }
    
    median_difference <- median(
      difference_data$defect_rate_difference
    )
    
    positive_count <- sum(
      difference_data$defect_rate_difference > 0
    )
    tied_count <- sum(
      difference_data$defect_rate_difference == 0
    )
    negative_count <- sum(
      difference_data$defect_rate_difference < 0
    )
    
    evidence_subtitle <- paste0(
      "Median increase = ",
      formatC(
        median_difference * 100,
        format = "f",
        digits = 2
      ),
      " percentage points; one-sided paired Wilcoxon ",
      format_p_value(
        paired_p_value
      ),
      ".\nPositive / tied / negative releases: ",
      positive_count,
      " / ",
      tied_count,
      " / ",
      negative_count,
      "."
    )
    
    ggplot(
      difference_data,
      aes(
        x = comparison,
        y = defect_rate_difference
      )
    ) +
      geom_hline(
        yintercept = 0,
        color = NEUTRAL_COLOR,
        linewidth = 0.75,
        linetype = "dashed"
      ) +
      geom_boxplot(
        width = 0.42,
        fill = "#F3F3F3",
        color = "#606060",
        linewidth = 0.65,
        outlier.shape = NA
      ) +
      geom_jitter(
        aes(
          color = direction
        ),
        width = 0.08,
        height = 0,
        size = 3.1,
        alpha = 0.88,
        show.legend = FALSE
      ) +
      geom_point(
        data =
          legend_direction_data,
        aes(
          x =
            comparison,
          y =
            defect_rate_difference,
          color =
            direction
        ),
        inherit.aes = FALSE,
        size = 0,
        alpha = 0,
        show.legend = TRUE
      ) +
      geom_errorbar(
        data = summary_data,
        aes(
          x = comparison,
          ymin = ci_lower,
          ymax = ci_upper
        ),
        inherit.aes = FALSE,
        width = 0.09,
        linewidth = 0.95,
        color = "black"
      ) +
      geom_point(
        data = summary_data,
        aes(
          x = comparison,
          y = mean_difference
        ),
        inherit.aes = FALSE,
        shape = 21,
        size = 5.0,
        fill = "white",
        color = "black",
        stroke = 1.15
      ) +
      scale_color_manual(
        values = c(
          "Positive" = HIT_COLOR,
          "Tied" = NEUTRAL_COLOR,
          "Negative" = NO_HIT_COLOR
        ),
        limits =
          direction_levels,
        breaks =
          direction_levels,
        labels = c(
          "Positive" = "Higher with token hit",
          "Tied" = "No difference",
          "Negative" = "Lower with token hit"
        ),
        drop = FALSE
      ) +
      scale_y_continuous(
        labels = label_percent(
          accuracy = 1
        ),
        expand = expansion(
          mult = c(
            0.08,
            0.12
          )
        )
      ) +
      labs(
        x = NULL,
        y = paste0(
          "Difference in defective-file rate\n",
          "(causal-token hit \u2212 no hit)"
        ),
        title = NULL,
        subtitle = NULL,
        caption = NULL,
        color = NULL
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
        panel.grid.major.x = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(
          color = "#D7D7D7",
          linewidth = 0.50
        ),
        plot.title = element_text(
          size = 20,
          face = "bold",
          color = "black",
          hjust = 0.5,
          margin = margin(
            b = 7
          )
        ),
        plot.subtitle = element_text(
          size = 11.5,
          color = "#2F2F2F",
          hjust = 0.5,
          lineheight = 1.12,
          margin = margin(
            b = 12
          )
        ),
        plot.caption = element_text(
          size = 9.5,
          color = "#4A4A4A",
          hjust = 0,
          lineheight = 1.08,
          margin = margin(
            t = 12
          )
        ),
        axis.title.y = element_text(
          size = 16,
          color = "black",
          lineheight = 1.05,
          margin = margin(
            r = 12
          )
        ),
        axis.text.x = element_text(
          size = 14,
          color = "black",
          margin = margin(
            t = 8
          )
        ),
        axis.text.y = element_text(
          size = 13,
          color = "black"
        ),
        legend.position = "bottom",
        legend.direction = "vertical",
        legend.box = "vertical",
        legend.justification = "center",
        legend.box.just = "center",
        legend.text = element_text(
          size = 9.0,
          color = "black"
        ),
        legend.key.width = grid::unit(
          0.38,
          "cm"
        ),
        legend.spacing.x = grid::unit(
          0.08,
          "cm"
        ),
        plot.margin = margin(
          t = 20,
          r = 24,
          b = 18,
          l = 24
        )
      ) +
      guides(
        color = guide_legend(
          title = NULL,
          ncol = 1,
          byrow = TRUE,
          override.aes = list(
            shape = c(
              16,
              16,
              16
            ),
            size = c(
              3.2,
              3.2,
              3.2
            ),
            alpha = c(
              1,
              1,
              1
            ),
            color = c(
              HIT_COLOR,
              NEUTRAL_COLOR,
              NO_HIT_COLOR
            )
          )
        )
      )
  }
  
  
  build_odds_ratio_plot <- function(
    release_statistics,
    pooled_tests,
    logistic_results
  ) {
    release_data <- release_statistics[
      is.finite(
        release_statistics$corrected_odds_ratio
      ) &
        release_statistics$corrected_odds_ratio > 0 &
        is.finite(
          release_statistics$corrected_ci_lower
        ) &
        is.finite(
          release_statistics$corrected_ci_upper
        ) &
        release_statistics$corrected_ci_lower > 0,
      ,
      drop = FALSE
    ]
    
    release_data <- release_data[
      order(
        release_data$corrected_odds_ratio,
        decreasing = FALSE
      ),
      ,
      drop = FALSE
    ]
    
    forest_data <- data.frame(
      display_label = paste0(
        release_data$project,
        ": ",
        release_data$test_release
      ),
      odds_ratio = release_data$corrected_odds_ratio,
      ci_lower = release_data$corrected_ci_lower,
      ci_upper = release_data$corrected_ci_upper,
      evidence_class = ifelse(
        release_data$corrected_ci_lower > 1,
        "Release CI above 1",
        ifelse(
          release_data$corrected_ci_upper < 1,
          "Release CI below 1",
          "Release CI includes 1"
        )
      ),
      is_summary = FALSE,
      stringsAsFactors = FALSE
    )
    
    cmh_row <- pooled_tests[
      pooled_tests$analysis ==
        "CMH stratified by release",
      ,
      drop = FALSE
    ]
    
    adjusted_row <- logistic_results[
      logistic_results$analysis ==
        "Defect association adjusted for file size and release",
      ,
      drop = FALSE
    ]
    
    summary_rows <- list()
    
    if (
      nrow(cmh_row) == 1 &&
      is.finite(cmh_row$estimate) &&
      is.finite(cmh_row$ci_lower) &&
      is.finite(cmh_row$ci_upper)
    ) {
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        display_label = "Overall: release-stratified CMH",
        odds_ratio = cmh_row$estimate,
        ci_lower = cmh_row$ci_lower,
        ci_upper = cmh_row$ci_upper,
        evidence_class = "Overall evidence",
        is_summary = TRUE,
        stringsAsFactors = FALSE
      )
    }
    
    if (
      nrow(adjusted_row) == 1 &&
      is.finite(adjusted_row$odds_ratio) &&
      is.finite(adjusted_row$ci_lower) &&
      is.finite(adjusted_row$ci_upper)
    ) {
      summary_rows[[length(summary_rows) + 1]] <- data.frame(
        display_label = "Overall: adjusted for file size and release",
        odds_ratio = adjusted_row$odds_ratio,
        ci_lower = adjusted_row$ci_lower,
        ci_upper = adjusted_row$ci_upper,
        evidence_class = "Overall evidence",
        is_summary = TRUE,
        stringsAsFactors = FALSE
      )
    }
    
    if (length(summary_rows) > 0) {
      forest_data <- rbind(
        forest_data,
        do.call(
          rbind,
          summary_rows
        )
      )
    }
    
    forest_data$display_label <- factor(
      forest_data$display_label,
      levels = forest_data$display_label
    )
    
    evidence_class_levels <- c(
      "Release CI above 1",
      "Release CI includes 1",
      "Release CI below 1",
      "Overall evidence"
    )
    
    forest_data$evidence_class <- factor(
      forest_data$evidence_class,
      levels =
        evidence_class_levels
    )
    

    legend_evidence_data <- data.frame(
      odds_ratio =
        rep(
          1,
          length(
            evidence_class_levels
          )
        ),
      display_label = factor(
        rep(
          as.character(
            forest_data$display_label[1]
          ),
          length(
            evidence_class_levels
          )
        ),
        levels =
          levels(
            forest_data$display_label
          )
      ),
      evidence_class = factor(
        evidence_class_levels,
        levels =
          evidence_class_levels
      ),
      stringsAsFactors = FALSE
    )
    
    cmh_text <- if (nrow(cmh_row) == 1) {
      paste0(
        "CMH OR = ",
        formatC(
          cmh_row$estimate,
          format = "f",
          digits = 2
        ),
        " [",
        formatC(
          cmh_row$ci_lower,
          format = "f",
          digits = 2
        ),
        ", ",
        formatC(
          cmh_row$ci_upper,
          format = "f",
          digits = 2
        ),
        "], ",
        format_p_value(
          cmh_row$p_value
        )
      )
    } else {
      "CMH result unavailable"
    }
    
    adjusted_text <- if (nrow(adjusted_row) == 1) {
      paste0(
        "Adjusted OR = ",
        formatC(
          adjusted_row$odds_ratio,
          format = "f",
          digits = 2
        ),
        " [",
        formatC(
          adjusted_row$ci_lower,
          format = "f",
          digits = 2
        ),
        ", ",
        formatC(
          adjusted_row$ci_upper,
          format = "f",
          digits = 2
        ),
        "], ",
        format_p_value(
          adjusted_row$p_value
        )
      )
    } else {
      "Adjusted model result unavailable"
    }
    
    ggplot(
      forest_data,
      aes(
        x = odds_ratio,
        y = display_label
      )
    ) +
      annotate(
        "rect",
        xmin = 1,
        xmax = Inf,
        ymin = -Inf,
        ymax = Inf,
        fill = HIT_COLOR,
        alpha = 0.035
      ) +
      geom_vline(
        xintercept = 1,
        color = NEUTRAL_COLOR,
        linewidth = 0.78,
        linetype = "dashed"
      ) +
      geom_segment(
        data = forest_data[
          !forest_data$is_summary,
          ,
          drop = FALSE
        ],
        aes(
          x = ci_lower,
          xend = ci_upper,
          y = display_label,
          yend = display_label
        ),
        linewidth = 0.68,
        color = "#787878"
      ) +
      geom_segment(
        data = forest_data[
          forest_data$is_summary,
          ,
          drop = FALSE
        ],
        aes(
          x = ci_lower,
          xend = ci_upper,
          y = display_label,
          yend = display_label
        ),
        linewidth = 1.15,
        color = SUMMARY_COLOR
      ) +
      geom_point(
        aes(
          fill = evidence_class,
          shape = is_summary,
          size = is_summary
        ),
        color = "black",
        stroke = 0.78,
        show.legend = FALSE
      ) +
      geom_point(
        data =
          legend_evidence_data,
        aes(
          x =
            odds_ratio,
          y =
            display_label,
          fill =
            evidence_class
        ),
        inherit.aes = FALSE,
        shape = 21,
        size = 0,
        alpha = 0,
        stroke = 0,
        show.legend = TRUE
      ) +
      scale_x_log10(
        breaks = c(
          0.05,
          0.1,
          0.25,
          0.5,
          1,
          2,
          4,
          10,
          25,
          100
        ),
        labels = label_number(
          accuracy = 0.01
        ),
        expand = expansion(
          mult = c(
            0.04,
            0.08
          )
        )
      ) +
      scale_fill_manual(
        values = c(
          "Release CI above 1" = HIT_COLOR,
          "Release CI includes 1" = "#D9D9D9",
          "Release CI below 1" = NO_HIT_COLOR,
          "Overall evidence" = "#FFFFFF"
        ),
        limits =
          evidence_class_levels,
        breaks =
          evidence_class_levels,
        drop = FALSE
      ) +
      scale_shape_manual(
        values = c(
          "FALSE" = 21,
          "TRUE" = 23
        )
      ) +
      scale_size_manual(
        values = c(
          "FALSE" = 2.9,
          "TRUE" = 5.0
        )
      ) +
      labs(
        x = paste0(
          "Odds ratio for defective file: ",
          "causal-token HIT vs. NO HIT"
        ),
        y = NULL,
        title = NULL,
        subtitle = NULL,
        caption = NULL,
        fill = NULL
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
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_line(
          color = "#D7D7D7",
          linewidth = 0.50
        ),
        plot.title = element_text(
          size = 19,
          face = "bold",
          color = "black",
          hjust = 0.5,
          margin = margin(
            b = 7
          )
        ),
        plot.subtitle = element_text(
          size = 11.2,
          color = "#2F2F2F",
          hjust = 0.5,
          lineheight = 1.12,
          margin = margin(
            b = 12
          )
        ),
        plot.caption = element_text(
          size = 9.2,
          color = "#4A4A4A",
          hjust = 0,
          lineheight = 1.08,
          margin = margin(
            t = 12
          )
        ),
        axis.title.x = element_text(
          size = 15,
          color = "black",
          hjust = 0.5,
          margin = margin(
            t = 10
          )
        ),
        axis.text.x = element_text(
          size = 11.5,
          color = "black"
        ),
        axis.text.y = element_text(
          size = 9.3,
          color = "black",
          face = ifelse(
            levels(
              forest_data$display_label
            ) %in% c(
              "Overall: release-stratified CMH",
              "Overall: adjusted for file size and release"
            ),
            "bold",
            "plain"
          )
        ),
        legend.position = "bottom",
        legend.text = element_text(
          size = 10.2,
          color = "black"
        ),
        legend.key.width = grid::unit(
          0.65,
          "cm"
        ),
        plot.margin = margin(
          t = 20,
          r = 24,
          b = 18,
          l = 24
        )
      ) +
      guides(
        fill = guide_legend(
          title = NULL,
          override.aes = list(
            shape = c(
              21,
              21,
              21,
              23
            ),
            size = c(
              3.2,
              3.2,
              3.2,
              4.2
            ),
            color = c(
              "black",
              "black",
              "black",
              "black"
            ),
            fill = c(
              HIT_COLOR,
              "#D9D9D9",
              NO_HIT_COLOR,
              "#FFFFFF"
            ),
            stroke = c(
              0.78,
              0.78,
              0.78,
              0.90
            ),
            alpha = c(
              1,
              1,
              1,
              1
            )
          )
        ),
        shape = "none",
        size = "none"
      )
  }
  
  
  dir.create(
    OUTPUT_ROOT,
    recursive = TRUE,
    showWarnings = FALSE
  )
  
  ranking_paths <- list.files(
    INPUT_ROOT,
    pattern = RANKING_FILE_PATTERN,
    full.names = TRUE,
    recursive = FALSE,
    ignore.case = TRUE
  )
  
  if (length(ranking_paths) == 0) {
    stop(
      paste0(
        "No full file-ranking CSV was found in ",
        INPUT_ROOT,
        ". Run evaluate_file_level_causal_token_effectiveness.py first."
      )
    )
  }
  
  ranking_paths <- sort(ranking_paths)
  
  all_file_rows <- lapply(
    ranking_paths,
    load_file_level_ranking
  )
  
  all_file_data <- do.call(
    rbind,
    all_file_rows
  )
  
  rownames(all_file_data) <- NULL
  
  release_names <- unique(
    all_file_data$test_release
  )
  
  release_statistics_rows <- lapply(
    release_names,
    function(release_name) {
      calculate_release_statistics(
        all_file_data[
          all_file_data$test_release == release_name,
          ,
          drop = FALSE
        ]
      )
    }
  )
  
  release_statistics <- do.call(
    rbind,
    release_statistics_rows
  )
  
  rownames(release_statistics) <- NULL
  
  pooled_a <- sum(
    all_file_data$token_hit == 1 &
      all_file_data$file_label == 1
  )
  pooled_b <- sum(
    all_file_data$token_hit == 1 &
      all_file_data$file_label == 0
  )
  pooled_c <- sum(
    all_file_data$token_hit == 0 &
      all_file_data$file_label == 1
  )
  pooled_d <- sum(
    all_file_data$token_hit == 0 &
      all_file_data$file_label == 0
  )
  
  pooled_contingency <- data.frame(
    token_status = c(
      "Causal-token hit",
      "Causal-token hit",
      "No causal-token hit",
      "No causal-token hit"
    ),
    file_status = c(
      "Defective",
      "Non-defective",
      "Defective",
      "Non-defective"
    ),
    count = c(
      pooled_a,
      pooled_b,
      pooled_c,
      pooled_d
    ),
    stringsAsFactors = FALSE
  )
  
  pooled_matrix <- matrix(
    c(
      pooled_a,
      pooled_b,
      pooled_c,
      pooled_d
    ),
    nrow = 2,
    byrow = TRUE,
    dimnames = list(
      token_hit = c("Yes", "No"),
      defective_file = c("Yes", "No")
    )
  )
  
  pooled_fisher_result <- fisher.test(
    pooled_matrix,
    alternative = "two.sided"
  )
  
  eligible_releases <- release_statistics$test_release[
    release_statistics$cmh_eligible
  ]
  
  cmh_estimate <- NA_real_
  cmh_ci_lower <- NA_real_
  cmh_ci_upper <- NA_real_
  cmh_p_value <- NA_real_
  
  if (length(eligible_releases) >= 2) {
    cmh_array <- build_cmh_array(
      all_file_data,
      eligible_releases
    )
    
    cmh_result <- mantelhaen.test(
      cmh_array,
      alternative = "two.sided",
      correct = FALSE
    )
    
    cmh_estimate <- unname(cmh_result$estimate)
    cmh_ci_lower <- cmh_result$conf.int[1]
    cmh_ci_upper <- cmh_result$conf.int[2]
    cmh_p_value <- cmh_result$p.value
  }
  
  paired_rate_data <- release_statistics[
    is.finite(release_statistics$token_hit_defect_rate) &
      is.finite(
        release_statistics$token_no_hit_defect_rate
      ),
    ,
    drop = FALSE
  ]
  
  paired_rate_result <- suppressWarnings(
    wilcox.test(
      paired_rate_data$token_hit_defect_rate,
      paired_rate_data$token_no_hit_defect_rate,
      paired = TRUE,
      alternative = "greater",
      exact = FALSE,
      conf.int = FALSE
    )
  )
  
  pooled_tests <- rbind(
    data.frame(
      analysis = paste0(
        "Pooled Fisher test without release ",
        "stratification"
      ),
      estimate_name = "odds_ratio",
      estimate = unname(pooled_fisher_result$estimate),
      ci_lower = pooled_fisher_result$conf.int[1],
      ci_upper = pooled_fisher_result$conf.int[2],
      p_value = pooled_fisher_result$p.value,
      release_count = length(release_names),
      file_count = nrow(all_file_data),
      interpretation = paste0(
        "Descriptive pooled association that ignores ",
        "release stratification."
      ),
      stringsAsFactors = FALSE
    ),
    data.frame(
      analysis = "CMH stratified by release",
      estimate_name = "common_odds_ratio",
      estimate = cmh_estimate,
      ci_lower = cmh_ci_lower,
      ci_upper = cmh_ci_upper,
      p_value = cmh_p_value,
      release_count = length(eligible_releases),
      file_count = sum(
        all_file_data$test_release %in%
          eligible_releases
      ),
      interpretation = paste0(
        "Primary association test controlling for ",
        "target-release strata."
      ),
      stringsAsFactors = FALSE
    ),
    data.frame(
      analysis = paste0(
        "Paired release-level defect-rate ",
        "comparison"
      ),
      estimate_name = "median_hit_minus_no_hit_rate",
      estimate = median(
        paired_rate_data$token_hit_defect_rate -
          paired_rate_data$token_no_hit_defect_rate
      ),
      ci_lower = NA_real_,
      ci_upper = NA_real_,
      p_value = paired_rate_result$p.value,
      release_count = nrow(paired_rate_data),
      file_count = nrow(all_file_data),
      interpretation = paste0(
        "Supplementary release-level comparison giving ",
        "equal weight to each eligible release."
      ),
      stringsAsFactors = FALSE
    )
  )
  
  all_file_data$release_factor <- factor(
    all_file_data$test_release
  )
  all_file_data$log2_line_count_plus_one <- log2(
    all_file_data$line_count + 1
  )
  
  defect_model <- suppressWarnings(
    glm(
      file_label ~
        token_hit +
        log2_line_count_plus_one +
        release_factor,
      data = all_file_data,
      family = binomial(link = "logit")
    )
  )
  
  size_bias_model <- suppressWarnings(
    glm(
      token_hit ~
        log2_line_count_plus_one +
        release_factor,
      data = all_file_data,
      family = binomial(link = "logit")
    )
  )
  
  logistic_results <- rbind(
    extract_logistic_term(
      model = defect_model,
      term_name = "token_hit",
      analysis_name = paste0(
        "Defect association adjusted for file size ",
        "and release"
      ),
      interpretation = paste0(
        "Odds ratio above 1 indicates that token-hit files ",
        "have higher defect odds after controlling for file ",
        "size and target release."
      )
    ),
    extract_logistic_term(
      model = size_bias_model,
      term_name = "log2_line_count_plus_one",
      analysis_name = "File-size bias in causal-token hit",
      interpretation = paste0(
        "Odds ratio above 1 indicates that a doubling of file ",
        "size is associated with a higher probability of ",
        "containing at least one file-level causal token."
      )
    )
  )
  
  write.csv(
    release_statistics,
    PER_RELEASE_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    pooled_contingency,
    POOLED_CONTINGENCY_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    pooled_tests,
    POOLED_TEST_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  write.csv(
    logistic_results,
    LOGISTIC_MODEL_OUTPUT_PATH,
    row.names = FALSE,
    fileEncoding = "UTF-8"
  )
  
  defect_rate_plot <- build_defect_rate_plot(
    release_statistics,
    pooled_tests
  )
  
  odds_ratio_plot <- build_odds_ratio_plot(
    release_statistics,
    pooled_tests,
    logistic_results
  )
  
  ggsave(
    filename = DEFECT_RATE_FIGURE_OUTPUT_PATH,
    plot = defect_rate_plot,
    width = DEFECT_RATE_FIGURE_WIDTH,
    height = DEFECT_RATE_FIGURE_HEIGHT,
    units = "in",
    dpi = OUTPUT_DPI,
    device = grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )
  
  ggsave(
    filename = ODDS_RATIO_FIGURE_OUTPUT_PATH,
    plot = odds_ratio_plot,
    width = ODDS_RATIO_FIGURE_WIDTH,
    height = ODDS_RATIO_FIGURE_HEIGHT,
    units = "in",
    dpi = OUTPUT_DPI,
    device = grDevices::cairo_pdf,
    family = "Arial",
    limitsize = FALSE
  )
  
  message(
    paste0(
      "Analyzed ",
      nrow(all_file_data),
      " files from ",
      length(release_names),
      " target releases."
    )
  )
  
  message(
    paste0(
      "Saved file-level causal-token discrimination results to ",
      OUTPUT_ROOT
    )
  )
}


for (
  aggregation_method in
  CLEAR_AGGREGATION_METHODS
) {
  run_line_level_causal_token_effectiveness(
    aggregation_method
  )
  
  run_file_level_causal_token_discrimination(
    aggregation_method
  )
}
