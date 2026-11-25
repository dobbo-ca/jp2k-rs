use std::fmt;

/// Every way a decode can fail. Go reports malformed input as free-form `fmt.Errorf` text; the
/// variants here group those messages, and the text is kept.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Malformed or corrupt input: a bad marker segment, box or packet header, a truncated stream.
    Invalid(String),
    /// Valid input that this decoder does not handle: RGN, palettes, ICC or sYCC colour, more
    /// than 4 components in `decode`, HT block coding, precision above 16 bits.
    Unsupported(String),
    /// A [`Limits`](crate::Limits) field, or a fixed sanity limit of the format, was exceeded.
    LimitExceeded(String),
    /// An allocation failed.
    OutOfMemory,
    /// The cancel flag was set.
    Cancelled,
    /// A reader failed (its `io::ErrorKind` and message).
    Io(std::io::ErrorKind, String),
    /// A panic was caught at an entry point: a bug, not a bad input. Go: `ErrInternal`.
    Internal(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(m) | Error::Unsupported(m) | Error::LimitExceeded(m) => {
                write!(f, "jp2k: {m}")
            }
            Error::OutOfMemory => f.write_str("jp2k: out of memory"),
            Error::Cancelled => f.write_str("jp2k: decode stopped: cancelled"),
            Error::Io(_, m) => write!(f, "jp2k: reading: {m}"),
            Error::Internal(m) => write!(f, "jp2k: internal error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.kind(), e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

macro_rules! invalid {
    ($($a:tt)*) => { $crate::Error::Invalid(format!($($a)*)) };
}
macro_rules! unsupported {
    ($($a:tt)*) => { $crate::Error::Unsupported(format!($($a)*)) };
}
macro_rules! limit {
    ($($a:tt)*) => { $crate::Error::LimitExceeded(format!($($a)*)) };
}
