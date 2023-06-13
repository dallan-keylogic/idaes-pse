#################################################################################
# The Institute for the Design of Advanced Energy Systems Integrated Platform
# Framework (IDAES IP) was produced under the DOE Institute for the
# Design of Advanced Energy Systems (IDAES).
#
# Copyright (c) 2018-2023 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory,
# National Technology & Engineering Solutions of Sandia, LLC, Carnegie Mellon
# University, West Virginia University Research Corporation, et al.
# All rights reserved.  Please see the files COPYRIGHT.md and LICENSE.md
# for full copyright and license information.
#################################################################################
"""
Tests for eNRTL example with aqueous NaCl and KCL

Reference:

[1] Local Composition Model for Excess Gibbs Energy of Electrolyte Systems, Pt 1.
Chen, C.-C., Britt, H.I., Boston, J.F., Evans, L.B.,
AIChE Journal, 1982, Vol. 28(4), pgs. 588-596

[2] Song, Y. and Chen, C.-C., Symmetric Electrolyte Nonrandom Two-Liquid Activity
Coefficient Model, Ind. Eng. Chem. Res., 2009, Vol. 48, pgs. 7788–7797

[3] New Data on Activity Coefficients of Potassium, Nitrate, and Chloride Ions
in Aqueous Solutions of KNO3 and KCl by Ion Selective Electrodes
Dash, D., Kumar, S., Mallika, C., Kamachi Mudali, U.,
ISRN Chemical Engineering, 2012, doi:10.5402/2012/730154

Figures digitized using WebPlotDigitizer, https://apps.automeris.io/wpd/,
May 2021

Authors: Andrew Lee, Douglas Allan
"""
import pytest
from copy import deepcopy

from pyomo.environ import ConcreteModel, value, units as pyunits

from idaes.models.properties.modular_properties.base.generic_property import (
    GenericParameterBlock,
)
from idaes.models.properties.modular_properties.examples.enrtl_H2O_NaCl_KCl import (
    configuration,
)
from idaes.models.properties.modular_properties.eos.enrtl_reference_states import InfiniteDilutionAqueous

# Data digitized from Fig 6 [1]
log_gamma_lc = {
    0.00964: 2.42954,
    0.04038: 2.42487,
    0.06763: 2.42318,
    0.09487: 2.41926,
    0.12210: 2.41031,
    0.14933: 2.40583,
    0.17656: 2.39521,
    0.20378: 2.38236,
    0.23100: 2.36671,
    0.25822: 2.35274,
    0.28543: 2.33487,
    0.31264: 2.31476,
    0.33984: 2.28795,
    0.36704: 2.26450,
    0.39423: 2.23323,
    0.42141: 2.19582,
    0.44858: 2.15562,
    0.47574: 2.11263,
    0.50290: 2.06462,
    0.53005: 2.00991,
    0.55718: 1.94851,
    0.58430: 1.87984,
    0.61140: 1.80114,
    0.63849: 1.71573,
    0.66556: 1.62363,
    0.69263: 1.52707,
    0.71968: 1.41878,
    0.74671: 1.30380,
    0.77374: 1.18771,
    0.80076: 1.06379,
    0.82531: 0.94403,
    0.85475: 0.79198,
    0.88051: 0.65576,
    0.90502: 0.51895,
    0.92955: 0.38644,
    0.95408: 0.25516,
    0.97367: 0.13637,
    0.98741: 0.06982,
}
log_gamma_pdh = {
    0.01088:-0.93415,
    0.03984:-0.93221,
    0.07135:-0.92886,
    0.09859:-0.92943,
    0.12585:-0.92647,
    0.15309:-0.92946,
    0.18034:-0.92613,
    0.20381:-0.92359,
    0.22666:-0.92180,
    0.24722:-0.92299,
    0.27446:-0.92421,
    0.30173:-0.91478,
    0.32899:-0.90341,
    0.35624:-0.90194,
    0.38350:-0.89861,
    0.41075:-0.89640,
    0.43801:-0.88693,
    0.46527:-0.87857,
    0.49252:-0.87636,
    0.51977:-0.87637,
    0.54703:-0.86746,
    0.57428:-0.86580,
    0.60153:-0.86024,
    0.62879:-0.85468,
    0.65605:-0.84800,
    0.68331:-0.84132,
    0.71057:-0.83185,
    0.73783:-0.82628,
    0.76510:-0.81346,
    0.79238:-0.79506,
    0.81908:-0.77986,
    0.84694:-0.75715,
    0.87424:-0.72480,
    0.90156:-0.68910,
    0.92890:-0.63889,
    0.95628:-0.56356,
    0.97632:-0.44123,
    0.98816:-0.31511,
    0.99677:-0.17118,
}
log_gamma_total = {
    0.01635:1.48219,
    0.04359:1.47826,
    0.07083:1.47657,
    0.09806:1.46707,
    0.12530:1.46203,
    0.15253:1.45197,
    0.17976:1.44748,
    0.20699:1.43686,
    0.23525:1.42489,
    0.26101:1.40985,
    0.28865:1.39161,
    0.31585:1.36927,
    0.34306:1.34972,
    0.37026:1.32682,
    0.39745:1.29499,
    0.42464:1.26316,
    0.45181:1.22352,
    0.47898:1.18277,
    0.50614:1.13475,
    0.53329:1.08228,
    0.56043:1.02478,
    0.58755:0.95947,
    0.61466:0.88355,
    0.64176:0.80429,
    0.66885:0.72056,
    0.69593:0.63180,
    0.72301:0.53859,
    0.75007:0.43812,
    0.77712:0.33486,
    0.80588:0.22640,
    0.82793:0.12620,
    0.84595:0.04934,
    0.87299:-0.06351,
    0.89251:-0.15164,
    0.91229:-0.24061,
    0.93937:-0.33383,
    0.96482:-0.36735,
    0.98260:-0.34992,
    0.99308:-0.25219,
    0.99746:-0.14346,
}
log_gamma_H2O = {
    0.01364:-1.33241,
    0.04095:-1.29656,
    0.06826:-1.26198,
    0.09558:-1.22070,
    0.12290:-1.18109,
    0.15022:-1.13925,
    0.17755:-1.09239,
    0.20329:-1.04816,
    0.23222:-0.99797,
    0.25955:-0.95264,
    0.28689:-0.90431,
    0.31421:-0.86286,
    0.34154:-0.81509,
    0.36888:-0.76640,
    0.39622:-0.71359,
    0.42355:-0.66673,
    0.45089:-0.61987,
    0.47822:-0.57245,
    0.50555:-0.52447,
    0.53289:-0.47761,
    0.56022:-0.43019,
    0.58755:-0.38444,
    0.61488:-0.33702,
    0.64221:-0.29574,
    0.66953:-0.25223,
    0.69685:-0.21653,
    0.72416:-0.17748,
    0.75147:-0.14569,
    0.77878:-0.11222,
    0.80779:-0.08209,
    0.83336:-0.06474,
    0.86065:-0.03609,
    0.88793:-0.02056,
    0.91520:-0.00850,
    0.94246:-0.00126,
    0.97105:0.00534,
    0.99208:0.00442,
}

class TestSymmetric_0KCl:
    # Test case for having parameters for a second salt with 0 concentration
    # Results should be the same as for the single salt case

    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        m.params = GenericParameterBlock(**configuration)

        m.state = m.params.build_state_block([1])

        # Need to set a value of T for checking expressions later
        m.state[1].temperature.set_value(298.15)

        # Set parameters to those used in 1982 paper
        m.params.Liq.tau["H2O", "Na+, Cl-"].set_value(8.885)
        m.params.Liq.tau["Na+, Cl-", "H2O"].set_value(-4.549)

        return m

    @pytest.mark.unit
    def test_parameters(self, model):
        assert model.params.Liq.tau["H2O", "Na+, Cl-"].value == 8.885
        assert model.params.Liq.tau["Na+, Cl-", "H2O"].value == -4.549

    @pytest.mark.unit
    def test_log_gamma_lc(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        for x, g in log_gamma_lc.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            # Need to correct for different reference state
            OFFSET = 2.42

            assert pytest.approx(g - OFFSET, abs=5e-2) == value(
                model.state[1].Liq_log_gamma_lc["Na+"]
            )
            assert pytest.approx(g - OFFSET, abs=5e-2) == value(
                model.state[1].Liq_log_gamma_lc["Cl-"]
            )

    @pytest.mark.unit
    def test_log_gamma_pdh(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        for x, g in log_gamma_pdh.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            # Need to correct for different reference state
            OFFSET = -0.92

            # Several lines come together at the edge of the plot
            # so loosen the tolerance to account for that
            if x > 0.98:
                tol = 8e-2
            else:
                tol = 5e-2

            assert pytest.approx(g - OFFSET, abs=tol) == value(
                model.state[1].Liq_log_gamma_pdh["Na+"]
            )
            assert pytest.approx(g - OFFSET, abs=tol) == value(
                model.state[1].Liq_log_gamma_pdh["Cl-"]
            )

    @pytest.mark.unit
    def test_log_gamma_born(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        for x, g in log_gamma_pdh.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            # No Born term should be present for symmetric reference state
            assert pytest.approx(0, abs=1e-12) == value(
                model.state[1].Liq_log_gamma_born["Na+"]
            )
            assert pytest.approx(0, abs=1e-12) == value(
                model.state[1].Liq_log_gamma_born["Cl-"]
            )

    @pytest.mark.unit
    def test_log_gamma_total(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        for x, g in log_gamma_total.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            # Need to correct for different reference state
            OFFSET = 1.47

            # Several lines come together at the edge of the plot
            # so loosen the tolerance to account for that
            if x > 0.98:
                tol = 8e-2
            else:
                tol = 5e-2

            assert pytest.approx(g - OFFSET, abs=tol) == value(
                model.state[1].Liq_log_gamma["Na+"]
            )
            assert pytest.approx(g - OFFSET, abs=tol) == value(
                model.state[1].Liq_log_gamma["Cl-"]
            )


    @pytest.mark.unit
    def test_log_gamma_H2O(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        # No offset for water
        OFFSET = 0

        for x, g in log_gamma_H2O.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            assert pytest.approx(g, abs=5e-2) == value(
                model.state[1].Liq_log_gamma["H2O"]
            )

class TestAqueous_0KCl:
    # Test case for having parameters for a second salt with 0 concentration
    # Results should be the same as for the single salt case
    # Ostensibly this has the same reference state as the literature source, but
    # I noticed a small offset when plotting data, probably from calibration of the
    # tool used to extract data from the plot
    OFFSET = 0.01
    @pytest.fixture(scope="class")
    def model(self):
        m = ConcreteModel()
        config = deepcopy(configuration)
        config["phases"]["Liq"]["equation_of_state_options"]["reference_state"] = InfiniteDilutionAqueous
        # Since the phase is water, these numbers don't actually matter, but they're taken from Table III from
        # Rashin and Honig, Reevaluation of the Born Model of Ion Hydration, J. Phys. Chem., 1985, 89, 5588-5593
        config["components"]["Na+"]["parameter_data"] = {"born_radius": (1.680, pyunits.angstrom)}
        config["components"]["K+"]["parameter_data"] = {"born_radius": (2.172, pyunits.angstrom)}
        config["components"]["Cl-"]["parameter_data"] = {"born_radius": (1.937, pyunits.angstrom)}
        m.params = GenericParameterBlock(**config)

        m.state = m.params.build_state_block([1])

        # Need to set a value of T for checking expressions later
        m.state[1].temperature.set_value(298.15)

        # Set parameters to those used in 1982 paper
        m.params.Liq.tau["H2O", "Na+, Cl-"].set_value(8.885)
        m.params.Liq.tau["Na+, Cl-", "H2O"].set_value(-4.549)

        return m

    @pytest.mark.unit
    def test_parameters(self, model):
        assert model.params.Liq.tau["H2O", "Na+, Cl-"].value == 8.885
        assert model.params.Liq.tau["Na+, Cl-", "H2O"].value == -4.549

    @pytest.mark.unit
    def test_log_gamma_lc(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        for x, g in log_gamma_lc.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            assert pytest.approx(g - self.OFFSET, abs=5e-2) == value(
                model.state[1].Liq_log_gamma_lc["Na+"]
            )
            assert pytest.approx(g - self.OFFSET, abs=5e-2) == value(
                model.state[1].Liq_log_gamma_lc["Cl-"]
            )

    @pytest.mark.unit
    def test_log_gamma_pdh(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        # Need to correct for different reference state
        OFFSET = 0

        for x, g in log_gamma_pdh.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)
            # Several lines come together at the edge of the plot
            # so loosen the tolerance to account for that
            if x > 0.98:
                tol = 8e-2
            else:
                tol = 5e-2

            assert pytest.approx(g - self.OFFSET, abs=tol) == value(
                model.state[1].Liq_log_gamma_pdh["Na+"]
            )
            assert pytest.approx(g - self.OFFSET, abs=tol) == value(
                model.state[1].Liq_log_gamma_pdh["Cl-"]
            )

    @pytest.mark.unit
    def test_log_gamma_born(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        for x, g in log_gamma_pdh.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            # Single solvent, Born term should be zero
            assert pytest.approx(0, abs=1e-12) == value(
                model.state[1].Liq_log_gamma_born["Na+"]
            )
            assert pytest.approx(0, abs=1e-12) == value(
                model.state[1].Liq_log_gamma_born["Cl-"]
            )

    @pytest.mark.unit
    def test_log_gamma_total(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)

        for x, g in log_gamma_total.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            # Several lines come together at the edge of the plot
            # so loosen the tolerance to account for that
            if x > 0.98:
                tol = 8e-2
            else:
                tol = 5e-2
            assert pytest.approx(g - self.OFFSET, abs=tol) == value(
                model.state[1].Liq_log_gamma["Na+"]
            )
            assert pytest.approx(g - self.OFFSET, abs=tol) == value(
                model.state[1].Liq_log_gamma["Cl-"]
            )

    @pytest.mark.unit
    def test_log_gamma_H2O(self, model):
        # Using 0 results in division by zero errors
        for k in model.state[1].mole_frac_phase_comp:
            model.state[1].mole_frac_phase_comp[k].set_value(1e-12)


        for x, g in log_gamma_H2O.items():
            model.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
            model.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
            model.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)

            assert pytest.approx(g - self.OFFSET, abs=5e-2) == value(
                model.state[1].Liq_log_gamma["H2O"]
            )

if __name__ == "__main__":
    m = ConcreteModel()
    config = deepcopy(configuration)
    # config["phases"]["Liq"]["equation_of_state_options"]["reference_state"] = InfiniteDilutionAqueous
    # Since the phase is water, these numbers don't actually matter, but they're taken from Table III from
    # Rashin and Honig, Reevaluation of the Born Model of Ion Hydration, J. Phys. Chem., 1985, 89, 5588-5593
    config["components"]["Na+"]["parameter_data"] = {"born_radius": (1.680, pyunits.angstrom)}
    config["components"]["K+"]["parameter_data"] = {"born_radius": (2.172, pyunits.angstrom)}
    config["components"]["Cl-"]["parameter_data"] = {"born_radius": (1.937, pyunits.angstrom)}
    m.params = GenericParameterBlock(**config)

    m.state = m.params.build_state_block([1])

    # Need to set a value of T for checking expressions later
    m.state[1].temperature.set_value(298.15)

    # Set parameters to those used in 1982 paper
    m.params.Liq.tau["H2O", "Na+, Cl-"].set_value(8.885)
    m.params.Liq.tau["Na+, Cl-", "H2O"].set_value(-4.549)

    m.state[1].mole_frac_phase_comp["Liq", "K+"].set_value(1e-12)

    log_gamma_pdh_vec = []

    for x, g in log_gamma_total.items():
        m.state[1].mole_frac_phase_comp["Liq", "H2O"].set_value(x)
        m.state[1].mole_frac_phase_comp["Liq", "Na+"].set_value((1 - x) / 2)
        m.state[1].mole_frac_phase_comp["Liq", "Cl-"].set_value((1 - x) / 2)
        # Several lines come together at the edge of the plot
        # so loosen the tolerance to account for that
        if x > 0.98:
            tol = 8e-2
        else:
            tol = 5e-2

        log_gamma_pdh_vec.append(value(
            m.state[1].Liq_log_gamma["Na+"]
        ))

    import numpy as np
    import matplotlib.pyplot as plt
    # Offset = 0.01
    Offset = 1.47
    x_vec = np.array([x for x in log_gamma_total.keys()])
    g_vec = np.array([g - Offset  for g in log_gamma_total.values()])
    log_gamma_pdh_vec = np.array(log_gamma_pdh_vec)

    plt.plot(x_vec, log_gamma_pdh_vec, x_vec, g_vec)
    plt.show()