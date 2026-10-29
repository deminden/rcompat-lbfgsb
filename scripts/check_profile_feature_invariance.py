#!/usr/bin/env python3
"""Verify explicit per-run profiles give identical results across Cargo feature settings.

Builds the dependency-free library in a temporary directory. This checks feature
isolation, not parity with an external R oracle. Run from any working directory.
"""
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
PROBE = r'''
use rcompat_lbfgsb::*;
fn main() {
    for profile in [ArithmeticProfile::R461X86, ArithmeticProfile::R461MacArm] {
        for numeric in [false,true] { for route in 0..3 {
            let mut bounds = Bounds::new(vec![-5.0;2],vec![5.0;2]).unwrap();
            if route == 1 { bounds.lower[0]=f64::NEG_INFINITY; }
            if route == 2 { bounds=Bounds::unbounded(2); }
            let f=|x:&[f64]|100.0*(x[1]-x[0]*x[0]).powi(2)+(1.0-x[0]).powi(2);
            let g=|x:&[f64]|vec![-400.0*x[0]*(x[1]-x[0]*x[0])-2.0*(1.0-x[0]),200.0*(x[1]-x[0]*x[0])];
            let c=OptimControl::default_for_dimension(2);
            let result=if numeric { optim_lbfgsb_with_profile(vec![-1.2,1.0],bounds,f,c,profile) }
                else { optim_lbfgsb_with_gradient_and_profile(vec![-1.2,1.0],bounds,f,g,c,profile) }.unwrap();
            println!("{} {numeric} {route} {:?} {:x} {:?} {} {}", profile.id(),
                result.par.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),result.value.to_bits(),result.counts,result.convergence,result.message);
        }}
    }
}
'''


def main():
    outputs = []
    with tempfile.TemporaryDirectory(prefix="rcompat-feature-check-") as temporary:
        directory = Path(temporary)
        probe = directory / "probe.rs"
        probe.write_text(PROBE)
        for arm_feature in [False, True]:
            library = directory / "librcompat_lbfgsb.rlib"
            executable = directory / "probe"
            cfg = ["--cfg", 'feature="arm-fma"'] if arm_feature else []
            subprocess.run(["rustc", "--edition=2021", "--crate-name", "rcompat_lbfgsb",
                            "--crate-type", "lib", "-O", *cfg, str(ROOT / "src/lib.rs"),
                            "-o", str(library)], check=True)
            subprocess.run(["rustc", "--edition=2021", "-O", str(probe), "--extern",
                            "rcompat_lbfgsb=" + str(library), "-o", str(executable)], check=True)
            outputs.append(subprocess.check_output([str(executable)], text=True))
    if outputs[0] != outputs[1]:
        raise SystemExit("Explicit profile results changed when arm-fma was enabled:\n"
                         + outputs[0] + "\narm-fma enabled:\n" + outputs[1])
    if len(outputs[0].splitlines()) != 12:
        raise SystemExit("Feature isolation probe did not execute all 12 cases")
    print("12 explicit profile/route/callback results are bit-identical across feature builds")


if __name__ == "__main__":
    main()
