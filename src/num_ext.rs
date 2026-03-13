use crate::{Error, Result};
use num_traits::{CheckedAdd, CheckedMul};

/// Checked arithmetic helpers that map overflow to crate `Error`.
pub(crate) trait CheckedFieldOps: Sized + Copy + CheckedAdd + CheckedMul {
    fn checked_add_checked(self, rhs: Self) -> Result<Self> {
        self.checked_add(&rhs).ok_or(Error::Overflow)
    }

    fn checked_mul_checked(self, rhs: Self) -> Result<Self> {
        self.checked_mul(&rhs).ok_or(Error::Overflow)
    }
}

impl<T> CheckedFieldOps for T where T: Sized + Copy + CheckedAdd + CheckedMul {}

/// Conversion helper that preserves conversion error sources through `Error`.
pub(crate) trait TryIntoChecked<T> {
    fn try_into_checked(self) -> Result<T>;
}

impl<T, U> TryIntoChecked<U> for T
where
    U: TryFrom<T>,
    Error: From<<U as TryFrom<T>>::Error>,
{
    fn try_into_checked(self) -> Result<U> {
        U::try_from(self).map_err(Error::from)
    }
}
