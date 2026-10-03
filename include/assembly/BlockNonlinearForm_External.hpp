// ============================================================================
// BlockNonlinearForm_External.hpp
//
// mfem::BlockNonlinearForm that only borrows its integrators. MFEM's
// NonlinearForm can do so through UseExternalIntegrators(), but
// BlockNonlinearForm has no such option and deletes every integrator added to
// it; this class leaves them to their owner, as GlobalAssembly_Mixed needs for
// the LocalAssembly_Mixed it owns. Assembly is unchanged.
//
// Author: Chongran Zhao
// Date: Oct. 3, 2026
// Email: chongran_zhao@brown.edu
// ============================================================================
#ifndef BLOCK_NONLINEAR_FORM_EXTERNAL_HPP
#define BLOCK_NONLINEAR_FORM_EXTERNAL_HPP

#include <mfem.hpp>

class BlockNonlinearForm_External : public mfem::BlockNonlinearForm
{
public:
   // Empties the lists of integrators before the base destructor, which would
   // otherwise delete them.
   ~BlockNonlinearForm_External() override
   {
      dnfi.SetSize(0);
      bnfi.SetSize(0);
      fnfi.SetSize(0);
      bfnfi.SetSize(0);
   }
};

#endif
