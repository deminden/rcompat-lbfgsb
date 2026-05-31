use std::error::Error;
use std::fmt;

/// Error returned when optimization inputs are invalid or evaluation fails.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimError {
    /// A vector or argument had the wrong length.
    DimensionMismatch {
        /// Name of the argument whose length was invalid.
        name: &'static str,
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// A control field had an invalid value.
    InvalidControl {
        /// Name of the invalid control field.
        field: &'static str,
        /// Human-readable reason.
        reason: String,
    },
    /// A bounds vector or bounds entry was invalid.
    InvalidBounds {
        /// Index of the invalid bound entry when applicable.
        index: Option<usize>,
        /// Lower bound value when applicable.
        lower: Option<f64>,
        /// Upper bound value when applicable.
        upper: Option<f64>,
        /// Human-readable reason.
        reason: String,
    },
    /// An initial parameter was not finite.
    NonFiniteInitialParameter {
        /// Index of the non-finite parameter.
        index: usize,
        /// Parameter value.
        value: f64,
    },
    /// An initial parameter was outside its bounds.
    InitialParameterOutOfBounds {
        /// Index of the out-of-bounds parameter.
        index: usize,
        /// Parameter value.
        value: f64,
        /// Lower bound.
        lower: f64,
        /// Upper bound.
        upper: f64,
    },
    /// The initial objective value was not finite.
    NonFiniteInitialValue {
        /// Objective value returned by the user closure.
        value: f64,
    },
    /// The objective returned a non-finite value during optimization.
    NonFiniteObjective {
        /// Objective value returned by the user closure.
        value: f64,
    },
    /// A supplied gradient returned an invalid value.
    InvalidGradient {
        /// Index of the invalid gradient element.
        index: Option<usize>,
        /// Gradient value when applicable.
        value: Option<f64>,
        /// Human-readable reason.
        reason: String,
    },
    /// The internal optimizer backend failed.
    BackendFailure {
        /// Backend failure message.
        message: String,
    },
}

impl fmt::Display for OptimError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "dimension mismatch for {name}: expected {expected}, got {actual}"
            ),
            Self::InvalidControl { field, reason } => {
                write!(f, "invalid control field {field}: {reason}")
            }
            Self::InvalidBounds {
                index,
                lower,
                upper,
                reason,
            } => match index {
                Some(index) => write!(
                    f,
                    "invalid bounds at index {index} (lower={lower:?}, upper={upper:?}): {reason}"
                ),
                None => write!(f, "invalid bounds: {reason}"),
            },
            Self::NonFiniteInitialParameter { index, value } => {
                write!(f, "initial parameter at index {index} is not finite: {value}")
            }
            Self::InitialParameterOutOfBounds {
                index,
                value,
                lower,
                upper,
            } => write!(
                f,
                "initial parameter at index {index} is outside bounds: {value} not in [{lower}, {upper}]"
            ),
            Self::NonFiniteInitialValue { value } => {
                write!(f, "initial objective value is not finite: {value}")
            }
            Self::NonFiniteObjective { value } => {
                write!(f, "objective returned a non-finite value: {value}")
            }
            Self::InvalidGradient {
                index,
                value,
                reason,
            } => match index {
                Some(index) => write!(
                    f,
                    "invalid gradient at index {index} (value={value:?}): {reason}"
                ),
                None => write!(f, "invalid gradient: {reason}"),
            },
            Self::BackendFailure { message } => write!(f, "backend failure: {message}"),
        }
    }
}

impl Error for OptimError {}
