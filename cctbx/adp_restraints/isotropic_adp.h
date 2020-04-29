#ifndef CCTBX_ADP_RESTRAINTS_ISOTROPIC_ADP_H
#define CCTBX_ADP_RESTRAINTS_ISOTROPIC_ADP_H

#include <cctbx/adp_restraints/adp_restraints.h>

namespace cctbx { namespace adp_restraints {

  struct isotropic_adp_proxy : public adp_restraint_proxy<1> {
    isotropic_adp_proxy() {}
    isotropic_adp_proxy(af::tiny<unsigned, 1> const &i_seqs_,
      double weight_)
    : adp_restraint_proxy<1>(i_seqs_, weight_)
    {}
  };

  class isotropic_adp : public adp_restraint_base_6<1> {
  public:
    isotropic_adp(
      scitbx::sym_mat3<double> const &u_cart,
      double weight)
    : adp_restraint_base_6<1>(af::tiny<bool, 1>(true), weight)
    {
      init_deltas(u_cart);
    }

    isotropic_adp(
      adp_restraint_params<double> const &params,
      isotropic_adp_proxy const &proxy)
    : adp_restraint_base_6<1>(params, proxy)
    {
      CCTBX_ASSERT(proxy.i_seqs[0] < params.u_cart.size());
      init_deltas(params.u_cart[proxy.i_seqs[0]]);
    }

    virtual
    void linearise(
      uctbx::unit_cell const &unit_cell,
      restraints::linearised_eqns_of_restraint<double> &linearised_eqns,
      xray::parameter_map<cctbx::xray::scatterer<double> > const &parameter_map,
      af::tiny<unsigned, 1> const &i_seqs)
    {
      linearise_1<isotropic_adp>(
        unit_cell, linearised_eqns, parameter_map, i_seqs[0], true, weight,
        deltas_.begin());
    }

    static double grad_u_iso(int) {
      CCTBX_NOT_IMPLEMENTED();
      return 1;
    }

    static const double* cart_grad_row(int i) {
      static const double grads_u_cart[6][6] = {
        { 2./3, -1./3, -1./3, 0, 0, 0},
        {-1./3,  2./3, -1./3, 0, 0, 0},
        {-1./3, -1./3,  2./3, 0, 0, 0},
        {    0,     0,     0, 1, 0, 0},
        {    0,     0,     0, 0, 1, 0},
        {    0,     0,     0, 0, 0, 1}
      };
      return &grads_u_cart[i][0];
    }

  protected:

    void init_deltas(scitbx::sym_mat3<double>const &u_cart) {
      double const u_iso = adptbx::u_cart_as_u_iso(u_cart);
      for (int i=0; i<6; i++)
        deltas_[i] = (i < 3 ? u_cart[i] - u_iso : u_cart[i]);
    }

  };


  // Polarization-anisotropic fp and fdp tensors have the same form as aadp
  // tensors, so their restraints also look very similar and can use most of
  // the same code.
  struct isotropic_fp_proxy : public isotropic_adp_proxy {
    isotropic_fp_proxy() {}
    isotropic_fp_proxy(af::tiny<unsigned, 1> const &i_seqs_,
      double weight_)
    : isotropic_adp_proxy(i_seqs_, weight_)
    {}
  
  };

  class isotropic_fp : public isotropic_adp {
  public:
    isotropic_fp(
      scitbx::sym_mat3<double> const &fp_cart,
      double weight)
    : isotropic_adp(fp_cart, weight)
    {}

    // When this is constructed, it should receive an adp_restraint_params
    // object holding fp_cart values in its u_cart data member.
    isotropic_fp(
      adp_restraint_params<double> const &params,
      isotropic_adp_proxy const &proxy)
    : isotropic_adp(params, proxy)
    {}

    void linearise(
      uctbx::unit_cell const &unit_cell,
      restraints::linearised_eqns_of_restraint<double> &linearised_eqns,
      xray::parameter_map<cctbx::xray::scatterer<double> > const &parameter_map,
      af::tiny<unsigned, 1> const &i_seqs)
    {
      linearise_1<isotropic_adp>(
        unit_cell, linearised_eqns, parameter_map, i_seqs[0], true, weight,
        deltas_.begin(), true, false);
    }

  };

  struct isotropic_fdp_proxy : public isotropic_adp_proxy {
    isotropic_fdp_proxy() {}
    isotropic_fdp_proxy(af::tiny<unsigned, 1> const &i_seqs_,
      double weight_)
    : isotropic_adp_proxy(i_seqs_, weight_)
    {}
  };

  class isotropic_fdp : public isotropic_adp {
  public:
    isotropic_fdp(
      scitbx::sym_mat3<double> const &fdp_cart,
      double weight)
    : isotropic_adp(fdp_cart, weight)
    {}

    // When this is constructed, it should receive an adp_restraint_params
    // object holding fdp_cart values in its u_cart data member.
    isotropic_fdp(
      adp_restraint_params<double> const &params,
      isotropic_adp_proxy const &proxy)
    : isotropic_adp(params, proxy)
    {}

    void linearise(
      uctbx::unit_cell const &unit_cell,
      restraints::linearised_eqns_of_restraint<double> &linearised_eqns,
      xray::parameter_map<cctbx::xray::scatterer<double> > const &parameter_map,
      af::tiny<unsigned, 1> const &i_seqs)
    {
      linearise_1<isotropic_adp>(
        unit_cell, linearised_eqns, parameter_map, i_seqs[0], true, weight,
        deltas_.begin(), false, true);
    }

  };

}} // cctbx::adp_restraints

#endif
