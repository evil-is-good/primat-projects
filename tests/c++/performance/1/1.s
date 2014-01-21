	.file	"1.cpp"
	.section	.rodata.str1.1,"aMS",@progbits,1
.LC2:
	.string	"%f\n"
	.text
	.globl	main
	.type	main, @function
main:
.LFB31:
	.cfi_startproc
	subq	$40, %rsp
	.cfi_def_cfa_offset 48
	movl	$0, %eax
	fldz
	flds	.LC1(%rip)
	movl	$1000000, %ecx
	fld	%st(0)
	fxch	%st(2)
.L4:
	fstl	A(,%rax,8)
	movq	%rax, 16(%rsp)
	fildq	16(%rsp)
	testq	%rax, %rax
	jns	.L2
	fadd	%st(2), %st
.L2:
	fstpl	B(,%rax,8)
	movq	%rcx, %rdx
	subq	%rax, %rdx
	movq	%rdx, 16(%rsp)
	fildq	16(%rsp)
	testq	%rdx, %rdx
	jns	.L3
	fadd	%st(3), %st
.L3:
	fstpl	C(,%rax,8)
	addq	$1, %rax
	cmpq	$1000000, %rax
	jne	.L4
	fstp	%st(0)
	fstp	%st(0)
	fstp	%st(0)
	movl	$1000, %edx
	jmp	.L5
.L6:
	fldl	B(,%rax,8)
	faddl	C(,%rax,8)
	fstpl	A(,%rax,8)
	addq	$1, %rax
	cmpq	$1000000, %rax
	jne	.L6
	subq	$1, %rdx
	je	.L7
.L5:
	movl	$0, %eax
	jmp	.L6
.L7:
	fldl	A+80(%rip)
	fstpl	(%rsp)
	movl	$.LC2, %esi
	movl	$1, %edi
	movl	$0, %eax
	call	__printf_chk
	movl	$0, %eax
	addq	$40, %rsp
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE31:
	.size	main, .-main
	.globl	C
	.bss
	.align 32
	.type	C, @object
	.size	C, 8000000
C:
	.zero	8000000
	.globl	B
	.align 32
	.type	B, @object
	.size	B, 8000000
B:
	.zero	8000000
	.globl	A
	.align 32
	.type	A, @object
	.size	A, 8000000
A:
	.zero	8000000
	.section	.rodata.cst4,"aM",@progbits,4
	.align 4
.LC1:
	.long	1602224128
	.ident	"GCC: (Ubuntu/Linaro 4.7.2-2ubuntu1) 4.7.2"
	.section	.note.GNU-stack,"",@progbits
