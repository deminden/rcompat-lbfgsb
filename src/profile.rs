/// Arithmetic target for one optimization run, independent of the host CPU.
///
/// Profiles model measured reference rounding; they do not change controls or
/// guarantee parity for arbitrary user callback implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithmeticProfile {
    /// Measured x86 / single-threaded OpenBLAS 0.3.32 reference arithmetic.
    X86OpenBlas,
    /// Measured Apple Silicon / OpenBLAS 0.3.34 Vortex reference arithmetic.
    AppleSiliconOpenBlas,
}

impl ArithmeticProfile {
    /// Profile used by the original entry points, including the legacy feature.
    pub const fn legacy() -> Self {
        if cfg!(feature = "arm-fma") {
            Self::AppleSiliconOpenBlas
        } else {
            Self::X86OpenBlas
        }
    }

    /// Stable reference identifier for recording experiment configuration.
    pub const fn id(self) -> &'static str {
        match self {
            Self::X86OpenBlas => "x86-openblas-0.3.32",
            Self::AppleSiliconOpenBlas => "apple-silicon-openblas-0.3.34",
        }
    }

    pub(crate) fn mul_add(self, a: f64, b: f64, c: f64) -> f64 {
        match self {
            Self::X86OpenBlas => a * b + c,
            Self::AppleSiliconOpenBlas => a.mul_add(b, c),
        }
    }

    pub(crate) fn indexed_dot(
        self,
        length: usize,
        mut pair: impl FnMut(usize) -> (f64, f64),
    ) -> f64 {
        match self {
            Self::X86OpenBlas => {
                let plain_end = length - length % 4;
                let mut sum = 0.0;
                for index in 0..plain_end {
                    let (a, b) = pair(index);
                    sum += a * b;
                }
                for index in plain_end..length {
                    let (a, b) = pair(index);
                    sum = a.mul_add(b, sum);
                }
                sum
            }
            Self::AppleSiliconOpenBlas => {
                // Public OpenBLAS probes show a 32-element block kernel:
                // sixteen fused lanes, reduced as paired vector additions.
                // Lengths below a full block retain sequential fused rounding.
                let prefix = length - length % 32;
                let mut sum = 0.0;
                if prefix > 0 {
                    let mut lanes = [0.0_f64; 16];
                    for index in 0..prefix {
                        let (a, b) = pair(index);
                        lanes[index % 16] = a.mul_add(b, lanes[index % 16]);
                    }
                    for stride in [2, 4, 8] {
                        for start in (0..16).step_by(2 * stride) {
                            lanes[start] += lanes[start + stride];
                            lanes[start + 1] += lanes[start + stride + 1];
                        }
                    }
                    sum = lanes[0] + lanes[1];
                }
                for index in prefix..length {
                    let (a, b) = pair(index);
                    sum = a.mul_add(b, sum);
                }
                sum
            }
        }
    }

    pub(crate) fn short_dot(self, left: &[f64], right: &[f64]) -> f64 {
        debug_assert_eq!(left.len(), right.len());
        self.indexed_dot(left.len(), |i| (left[i], right[i]))
    }

    pub(crate) fn dot(self, exact: bool, left: &[f64], right: &[f64]) -> f64 {
        if exact {
            self.short_dot(left, right)
        } else {
            left.iter().zip(right).map(|(a, b)| a * b).sum()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_follow_independently_known_binary64_rounding() {
        let a = 1.0 + 2.0_f64.powi(-27);
        let b = 1.0 - 2.0_f64.powi(-27);
        assert_eq!(ArithmeticProfile::X86OpenBlas.mul_add(a, b, -1.0), 0.0);
        assert_eq!(
            ArithmeticProfile::AppleSiliconOpenBlas.mul_add(a, b, -1.0),
            -2.0_f64.powi(-54)
        );
        let left = [-1.0, a, 0.0, 0.0];
        let right = [1.0, b, 0.0, 0.0];
        assert_eq!(ArithmeticProfile::X86OpenBlas.short_dot(&left, &right), 0.0);
        assert_eq!(
            ArithmeticProfile::AppleSiliconOpenBlas.short_dot(&left, &right),
            -2.0_f64.powi(-54)
        );
    }
}

#[cfg(test)]
mod reference_dot_tests {
    use super::*;
    fn sample(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 32) as i64 - 2147483648) as f64 / ((*state & 1023) + 1) as f64
    }
    #[test]
    fn arm_indexed_and_contiguous_dots_match_200_independent_blas_calls() {
        let mut checked = 0;
        for line in include_str!("../fixtures/arithmetic/arm_dot.csv")
            .lines()
            .skip(1)
        {
            let row: Vec<_> = line.split(',').collect();
            assert_eq!(row.len(), 3);
            let n: usize = row[0].parse().unwrap();
            let mut state: u64 = row[1].parse().unwrap();
            let expected = u64::from_str_radix(row[2], 16).unwrap();
            let mut x = Vec::new();
            let mut y = Vec::new();
            for _ in 0..n {
                x.push(sample(&mut state));
                y.push(sample(&mut state));
            }
            let contiguous = ArithmeticProfile::AppleSiliconOpenBlas.short_dot(&x, &y);
            let mut visited = 0;
            let indexed = ArithmeticProfile::AppleSiliconOpenBlas.indexed_dot(n, |i| {
                assert_eq!(i, visited);
                visited += 1;
                (x[i], y[i])
            });
            assert_eq!(visited, n);
            assert_eq!(contiguous.to_bits(), expected, "length={n} seed={}", row[1]);
            assert_eq!(
                indexed.to_bits(),
                expected,
                "indexed length={n} seed={}",
                row[1]
            );
            checked += 1;
        }
        assert_eq!(checked, 200);
    }
}
