	.file	"3.cpp"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%lf %ld %ld\n"
	.section	.text.startup,"ax",@progbits
	.p2align 5,,31
	.globl	main
	.type	main, @function
main:
.LFB46:
	.cfi_startproc
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	movl	$10, %edx
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rsi, %rbx
	xorl	%esi, %esi
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	movq	(%rbx), %rdi
	call	strtol
	movq	8(%rbx), %rdi
	movl	$10, %edx
	xorl	%esi, %esi
	movq	%rax, %r12
	call	strtol
	movq	16(%rbx), %rdi
	movq	%rax, %rbp
	movl	$10, %edx
	xorl	%esi, %esi
	imull	%r12d, %ebp
	call	strtol
	xorl	%edi, %edi
	movq	%rax, %rbx
	call	time
	xorl	%edi, %edi
	movq	%rax, %r13
	call	time
	leal	0(%rbp,%rbx), %edx
	subq	%r13, %rax
	xorpd	%xmm0, %xmm0
	movq	%rax, %rcx
	movslq	%edx, %rdx
	movl	$.LC0, %esi
	movl	$1, %edi
	movl	$1, %eax
	call	__printf_chk
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE46:
	.size	main, .-main
	.ident	"GCC: (Ubuntu/Linaro 4.6.1-9ubuntu3) 4.6.1"
	.section	.note.GNU-stack,"",@progbits
