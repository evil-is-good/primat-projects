	.file	"1.cpp"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC0:
	.string	"%ld"
	.section	.text.startup,"ax",@progbits
	.p2align 4,,15
	.globl	main
	.type	main, @function
main:
.LFB31:
	.cfi_startproc
	subq	$2056, %rsp
	.cfi_def_cfa_offset 2064
	xorl	%edx, %edx
	movl	$3988292384, %eax
	.p2align 4,,10
	.p2align 3
.L18:
	movq	%rdx, %rcx
	shrq	%rcx
	movq	%rcx, %rsi
	xorq	%rax, %rsi
	testb	$1, %dl
	cmove	%rcx, %rsi
	movq	%rsi, %rcx
	shrq	%rcx
	movq	%rcx, %rdi
	xorq	%rax, %rdi
	andl	$1, %esi
	cmove	%rcx, %rdi
	movq	%rdi, %rcx
	shrq	%rcx
	movq	%rcx, %rsi
	xorq	%rax, %rsi
	andl	$1, %edi
	cmove	%rcx, %rsi
	movq	%rsi, %rcx
	shrq	%rcx
	movq	%rcx, %rdi
	xorq	%rax, %rdi
	andl	$1, %esi
	cmove	%rcx, %rdi
	movq	%rdi, %rcx
	shrq	%rcx
	movq	%rcx, %rsi
	xorq	%rax, %rsi
	andl	$1, %edi
	cmove	%rcx, %rsi
	movq	%rsi, %rcx
	shrq	%rcx
	movq	%rcx, %rdi
	xorq	%rax, %rdi
	andl	$1, %esi
	cmove	%rcx, %rdi
	movq	%rdi, %rcx
	shrq	%rcx
	movq	%rcx, %rsi
	xorq	%rax, %rsi
	andl	$1, %edi
	cmove	%rcx, %rsi
	movq	%rsi, %rcx
	shrq	%rcx
	movq	%rcx, %rdi
	xorq	%rax, %rdi
	andl	$1, %esi
	cmovne	%rdi, %rcx
	movq	%rcx, (%rsp,%rdx,8)
	addq	$1, %rdx
	cmpq	$256, %rdx
	jne	.L18
	movq	80(%rsp), %rdx
	movl	$.LC0, %esi
	movl	$1, %edi
	xorl	%eax, %eax
	call	__printf_chk
	xorl	%eax, %eax
	addq	$2056, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE31:
	.size	main, .-main
	.ident	"GCC: (Ubuntu/Linaro 4.6.1-9ubuntu3) 4.6.1"
	.section	.note.GNU-stack,"",@progbits
