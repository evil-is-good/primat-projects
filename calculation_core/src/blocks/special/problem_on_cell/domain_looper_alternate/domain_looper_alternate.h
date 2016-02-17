#ifndef DOMAIN_LOOPER_ALTERNATE
#define DOMAIN_LOOPER_ALTERNATE 1

template<st dim>
void loop_domain (const dealii::DoFHandler<dim> &dof_h,
        OnCell::BlackOnWhiteSubstituter &bows,
        dealii::CompressedSparsityPattern &csp)
{
};

#endif
