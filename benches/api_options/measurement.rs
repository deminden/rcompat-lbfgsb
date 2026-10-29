use super::*;
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Complete {
    pub(super) x: Vec<u64>,
    pub(super) value: u64,
    pub(super) counts: OptimCounts,
    pub(super) convergence: i32,
    pub(super) message: String,
    pub(super) iterations: usize,
    pub(super) reason: TerminationReason,
}

impl Complete {
    pub(super) fn fingerprint(&self) -> u64 {
        fingerprint::result_fingerprint(
            self.x.iter().copied().chain([
                self.value,
                self.counts.function as u64,
                self.counts.gradient as u64,
                self.convergence as u64,
                self.iterations as u64,
            ]),
            &self.message,
        )
    }
}

impl From<BackendResult> for Complete {
    fn from(r: BackendResult) -> Self {
        Self {
            x: r.x.iter().map(|v| v.to_bits()).collect(),
            value: r.value.to_bits(),
            counts: r.counts,
            convergence: r.convergence,
            message: r.message,
            iterations: r.iterations,
            reason: r.reason,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct Cost {
    pub(super) allocations: usize,
    pub(super) bytes: usize,
    pub(super) exponentials: usize,
}

pub(super) fn measure<P: BackendProblem>(
    problem: &mut P,
    initial: &[f64],
    bounds: &Bounds,
    control: BackendControl,
    model: &Model,
) -> (Complete, Cost, u128) {
    model.exponentials.set(0);
    let region = Region::new(GLOBAL);
    let start = Instant::now();
    let result = NativeBackend
        .minimize(problem, initial, &bounds.lower, &bounds.upper, control)
        .unwrap();
    let elapsed = start.elapsed().as_nanos();
    let stats = region.change();
    (
        result.into(),
        Cost {
            allocations: stats.allocations,
            bytes: stats.bytes_allocated,
            exponentials: model.exponentials.get(),
        },
        elapsed,
    )
}

#[derive(Clone, Copy)]
pub(super) struct Setup<'a> {
    pub(super) initial: &'a [f64],
    pub(super) bounds: &'a Bounds,
    pub(super) control: BackendControl,
    pub(super) model: &'a Model,
    pub(super) samples: usize,
    pub(super) stationary: bool,
}

impl Setup<'_> {
    pub(super) fn record<P: BackendProblem>(
        &self,
        problem: &mut P,
        expected: &mut Option<Complete>,
        mode: &str,
    ) {
        let Self {
            initial,
            bounds,
            control,
            model,
            samples,
            stationary,
        } = *self;
        let (warm, cost, _) = measure(problem, initial, bounds, control, model);
        if let Some(reference) = expected {
            assert_eq!(
                reference, &warm,
                "callback prototype changed complete optimizer results"
            );
        } else {
            *expected = Some(warm);
        }
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let (result, actual, elapsed) = measure(problem, initial, bounds, control, model);
            assert_eq!(&result, expected.as_ref().unwrap());
            assert_eq!(actual, cost);
            times.push(elapsed);
        }
        times.sort_unstable();
        let result = expected.as_ref().unwrap();
        println!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{:016x}",
            initial.len(),
            control.lmm,
            control.profile.id(),
            model.work,
            stationary,
            mode,
            times[samples / 2],
            cost.allocations,
            cost.bytes,
            cost.exponentials,
            result.counts.function,
            result.counts.gradient,
            result.iterations,
            result.fingerprint()
        );
    }
}
