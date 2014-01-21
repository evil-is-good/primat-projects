	.file	"1.cpp"
	.section	.text._Z5powerdl,"axG",@progbits,_Z5powerdl,comdat
	.weak	_Z5powerdl
	.type	_Z5powerdl, @function
_Z5powerdl:
.LFB0:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movsd	%xmm0, -8(%rbp)
	movq	%rdi, -16(%rbp)
	cmpq	$0, -16(%rbp)
	jle	.L2
	movq	-16(%rbp), %rax
	leaq	-1(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rdi
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	call	_Z5powerdl
	movsd	%xmm0, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	movapd	%xmm0, %xmm1
	mulsd	-8(%rbp), %xmm1
	movsd	%xmm1, -24(%rbp)
	movq	-24(%rbp), %rax
	jmp	.L3
.L2:
	movabsq	$4607182418800017408, %rax
.L3:
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	_Z5powerdl, .-_Z5powerdl
	.section	.text._Z7to_zerod,"axG",@progbits,_Z7to_zerod,comdat
	.weak	_Z7to_zerod
	.type	_Z7to_zerod, @function
_Z7to_zerod:
.LFB1:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movsd	%xmm0, -8(%rbp)
	movsd	-8(%rbp), %xmm0
	xorpd	%xmm1, %xmm1
	ucomisd	%xmm1, %xmm0
	jbe	.L11
.L10:
	movsd	-8(%rbp), %xmm0
	movsd	.LC0(%rip), %xmm1
	subsd	%xmm1, %xmm0
	call	_Z7to_zerod
	movsd	%xmm0, -16(%rbp)
	movq	-16(%rbp), %rax
	jmp	.L8
.L11:
	movl	$0, %eax
.L8:
	movq	%rax, -16(%rbp)
	movsd	-16(%rbp), %xmm0
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	_Z7to_zerod, .-_Z7to_zerod
	.section	.rodata
.LC3:
	.string	"%lf\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	$20, -8(%rbp)
	movabsq	$4611686018427387904, %rax
	movl	$20, %edi
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	call	_Z5powerdl
	movsd	%xmm0, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	movl	$.LC3, %edi
	movl	$1, %eax
	call	printf
	movabsq	$4611686018427387904, %rax
	movl	$20, %edi
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	call	_Z5powerdl
	movsd	%xmm0, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	call	_Z7to_zerod
	movsd	%xmm0, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -24(%rbp)
	movsd	-24(%rbp), %xmm0
	movl	$.LC3, %edi
	movl	$1, %eax
	call	printf
	movl	$0, %eax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
	.align 8
.LC0:
	.long	0
	.long	1072693248
	.ident	"GCC: (Ubuntu/Linaro 4.7.2-2ubuntu1) 4.7.2"
	.section	.note.GNU-stack,"",@progbits
