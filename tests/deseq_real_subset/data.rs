use super::support::fixtures::{RecordedControl as DeseqControl, RecordedResult as DeseqResult};
use super::*;

#[derive(Debug, Deserialize)]
pub(super) struct DeseqFixture {
    pub(super) fixture: String,
    pub(super) samples: usize,
    pub(super) coefficients: usize,
    pub(super) cases: Vec<DeseqCase>,
}

#[derive(Debug, Deserialize)]
pub(super) struct DeseqCase {
    pub(super) fixture: String,
    pub(super) gene: String,
    pub(super) dispersion: f64,
    pub(super) initial_par: Vec<f64>,
    pub(super) lower: Vec<f64>,
    pub(super) upper: Vec<f64>,
    pub(super) control: DeseqControl,
    pub(super) result: DeseqResult,
}

#[derive(Debug)]
pub(super) struct Table {
    pub(super) headers: Vec<String>,
    pub(super) rows: Vec<Vec<String>>,
}

impl Table {
    pub(super) fn column(&self, name: &str) -> usize {
        self.headers
            .iter()
            .position(|header| header == name)
            .unwrap_or_else(|| panic!("missing column {name}"))
    }

    pub(super) fn rows_by_gene<'a>(&'a self, gene: &str) -> Vec<&'a [String]> {
        let gene_column = self.column("gene");
        self.rows
            .iter()
            .filter(|row| row[gene_column] == gene)
            .map(Vec::as_slice)
            .collect()
    }

    pub(super) fn single_row_by_gene<'a>(&'a self, gene: &str) -> &'a [String] {
        let rows = self.rows_by_gene(gene);
        assert_eq!(rows.len(), 1, "expected one row for {gene}");
        rows[0]
    }
}

pub(super) fn deseq_fixture() -> DeseqFixture {
    serde_json::from_str(include_str!(
        "../../fixtures/deseq_real_subset/optim_cases.json"
    ))
    .unwrap()
}

pub(super) fn deseq_tables() -> DeseqTables {
    DeseqTables {
        gene_cases: parse_tsv(include_str!(
            "../../fixtures/deseq_real_subset/gene_cases.tsv"
        )),
        counts: parse_tsv(include_str!("../../fixtures/deseq_real_subset/counts.tsv")),
        coefficients: parse_tsv(include_str!(
            "../../fixtures/deseq_real_subset/coefficients_long.tsv"
        )),
        design: parse_tsv(include_str!(
            "../../fixtures/deseq_real_subset/design_matrix.tsv"
        )),
        size_factors: parse_tsv(include_str!(
            "../../fixtures/deseq_real_subset/size_factors.tsv"
        )),
    }
}

pub(super) struct DeseqTables {
    pub(super) gene_cases: Table,
    pub(super) counts: Table,
    pub(super) coefficients: Table,
    pub(super) design: Table,
    pub(super) size_factors: Table,
}

pub(super) fn parse_tsv(text: &str) -> Table {
    let mut lines = text.lines();
    let headers = lines
        .next()
        .expect("table header")
        .split('\t')
        .map(str::to_string)
        .collect();
    let rows = lines
        .filter(|line| !line.trim().is_empty())
        .map(|line| line.split('\t').map(str::to_string).collect())
        .collect();
    Table { headers, rows }
}

pub(super) fn initial_parameters_from_coefficients(coefficients: &Table, gene: &str) -> Vec<f64> {
    let gene_column = coefficients.column("gene");
    let index_column = coefficients.column("coefficient_index_1based");
    let beta_column = coefficients.column("betaNoOptim");
    let mut indexed = coefficients
        .rows
        .iter()
        .filter(|row| row[gene_column] == gene)
        .map(|row| {
            (
                row[index_column].parse::<usize>().unwrap(),
                parse_f64(&row[beta_column]),
            )
        })
        .collect::<Vec<_>>();
    indexed.sort_by_key(|(index, _)| *index);
    indexed.into_iter().map(|(_, value)| value).collect()
}

pub(super) fn design_matrix(design: &Table) -> Vec<Vec<f64>> {
    design
        .rows
        .iter()
        .map(|row| row[1..].iter().map(|value| parse_f64(value)).collect())
        .collect()
}

pub(super) fn size_factors(size_factors: &Table) -> Vec<f64> {
    size_factors
        .rows
        .iter()
        .map(|row| parse_f64(&row[1]))
        .collect()
}

pub(super) fn counts_for_gene(counts: &Table, gene: &str) -> Vec<f64> {
    counts.single_row_by_gene(gene)[1..]
        .iter()
        .map(|value| parse_f64(value))
        .collect()
}

pub(super) fn control_from_case(control: &DeseqControl) -> OptimControl {
    control.to_control(0, 10)
}

pub(super) fn parse_f64(value: &str) -> f64 {
    value.parse::<f64>().unwrap()
}
