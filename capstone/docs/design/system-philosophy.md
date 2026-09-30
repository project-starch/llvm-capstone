# Capstone system philosophy and protection model

Status: SYSTEM DIRECTION, 2026-09-30. This is the central statement of the
system's security goals and trust boundaries. It defines what implementations
must establish; it is not a claim that every goal is already qualified.

## System goal

An application should be able to use Linux services without granting Linux or
the monitor access to its private memory. Memory errors should be contained by
enforced object and domain boundaries.

We design against a malicious operating system and a malicious monitor, and
against memory errors inside applications. Linux supplies operating-system
services and backing storage. The monitor manages grants, scheduling and
revocation. Hardware constrains the authority that either can exercise. The
domain derives its pointers, manages object lifetimes and decides what data
to exchange with the outside world.

Management authority must remain distinct from data authority. Holding a
revocation handle, allocating backing or running in M-mode must not by itself
permit reading or modifying delegated private contents. The
[mapping candidate](caplified-mapping-tables.md#3-security-goal-and-trust)
applies this rule to translated memory; the
[physical-grant design](../plans/delegation-memory.md) applies it to physical
ownership. The [delegated ABI](../plans/delegation-abi.md) supplies Linux services
without putting a second operating system inside each domain.

## Attacker model

After a correctly initialized start, assume that Linux, its module and launcher,
the monitor, and other domains may be fully compromised and may cooperate.
They may execute arbitrary code, misuse every capability they can reach, choose
hostile instruction orderings, race operations, revoke storage, withhold service
and supply false syscall results. A memory-corruption exploit in the monitor
is included in this assumption; monitor memory safety is not a prerequisite for
excluding it from private-memory confidentiality and integrity trust.

Arbitrary code execution remains subject to the hardware's capability rules.
The attacker may attempt to forge or widen authority, but successful forgery,
unauthorized tag creation or bypass of access checks violates the hardware assumption.
Capabilities must constrain privileged execution paths as well as ordinary
loads and stores. An ordinary Linux mapping is not authority to access a frame
whose exclusive ownership has been transferred.

The protected application may contain memory errors. A fully compromised domain
can misuse its own reachable authority and disclose its own secrets; the
guarantee for other domains must survive. Protecting an object from a malicious
component within the same domain additionally requires keeping the object's
authority out of that component's reach.

## Protection boundaries

| Boundary | Threat | Required protection |
|---|---|---|
| Domain and external managers | Arbitrary code in Linux, the monitor or another domain | No private read, write or authority expansion without a matching data capability. For PRIVATE mappings, management cannot substitute other contents beneath live pointers or duplicate exclusive authority. Revocation may withdraw access. |
| Objects within a domain | Out-of-bounds access, invalid pointers and use-after-free | Accesses obey capability provenance, bounds, permissions and lifetime. Compiler and allocator rules must establish the intended object boundaries and revoke stale authority before reuse. |
| Syscall interface | Malformed, inconsistent, changing or deliberately false replies | Replies cannot manufacture capabilities, widen authority or cause unsafe memory operations in the domain runtime. External data remains untrusted input. |

Hardware checks the bounds and rights actually carried by a capability. It
cannot infer the programmer's intended object from an overly broad capability.
Object-level claims therefore require appropriate compiler and allocator
discipline. A wrong write within valid bounds, misuse of an accessible genuine
capability, or an application logic error is not automatically prevented by
memory safety. Coverage must name the object kinds and lifetime paths checked.

## Trust is specific to the property

| Property | Trust required |
|---|---|
| Private-memory confidentiality and integrity during execution | Correct hardware enforcement and the domain components that legitimately hold data authority or decide its release. Monitor and Linux correctness are not assumptions for excluding their unauthorized accesses. |
| Object-level memory safety inside a domain | Hardware plus the relevant compiler lowering, libc and allocator or Sublet rules for bounds, provenance and lifetime. This does not grant those components authority over other domains. |
| Register confidentiality across traps and domain switches | Hardware context protection. Monitor handling must receive a protected continuation, such as a sealed return capability, without gaining access to the saved private registers. Each delivery path needs qualification. |
| Syscall boundary safety | The domain-side marshalling and reply validation that hold private and exchange capabilities. Validation by an untrusted launcher is insufficient for this property. |
| Initial code, data and authority | A correct loader and setup, including the firmware involved in launch. Runtime isolation alone does not authenticate the initial program or capability distribution. |
| Availability and progress | Depend on Linux and the monitor. They may deny backing, revoke, stall or refuse scheduling; availability against them is not promised. |

The domain's own trusted computing base depends on which components hold which
authority. A label such as "firmware plus hardware is the TCB" loses these
property-specific distinctions. Conversely, excluding the monitor from runtime
data protection does not exclude it from the current startup assumption.
Measured and authenticated launch would require a separate design binding the
evidence to the actual execution and authority state; it is outside the present
scope.

## Syscall replies are adversarial inputs

Safe handling of malicious replies is part of the system goal. For example,
if `read(fd, buf, 16)` reports 4096 bytes, the domain runtime must reject or
safely fail that result rather than copy 4096 bytes. A returned integer address
must not become a data capability without an independently authorized grant.
Lengths, offsets, nested structures and state transitions must be checked
against domain-owned request state. Data checked in a shared buffer must not
become a different unchecked value before use.

This is the boundary addressed by
[Iago attacks](https://doi.org/10.1145/2451116.2451145): a malicious kernel can
use syscall results to make protected code act against its interests. Hardware
isolation does not remove the need for a safe syscall interface.

Plausibility checks cannot establish the truth of otherwise valid external
data. Linux can supply a different file, an attacker-selected key or a false
claim that a write reached durable storage. Applications that require data
authenticity, freshness or durability need the corresponding protocol and
independent trust anchors. They cannot obtain those guarantees merely by
checking buffer bounds. No transparent guarantee that arbitrary Linux
applications behave correctly under a lying kernel is made.

Private contents explicitly copied into an exchange buffer are disclosed to its
authorized readers. The application must decide what may leave the domain and
authenticate external inputs where its security policy requires it. Memory
isolation alone is not an end-to-end information-flow guarantee.

## Evidence and remaining scope

The [executable mapping model](../../tests/mapping-model/README.md) supplies
bounded evidence for the mapping and reclamation contract. It does not model
arbitrary syscall semantics or establish a complete application-security proof.
A systematic contract and adversarial campaign for syscall replies remain a
separate qualification obligation, even where individual validation checks or
functional tests already exist.

The [mapping design's qualification limits](caplified-mapping-tables.md#3-security-goal-and-trust)
remain in force: Linux-side isolation on the affected RTL waits for R-44, and
claims about shared, possibly tagged memory on the affected QEMU wait for the
Linux-store tag fix. Functional success on one platform does not discharge a
different platform's security assumptions.

Availability against the managers, physical fault injection, and comprehensive
side-channel resistance are not established by this protection model. Protected
storage, authenticated launch and later memory features need their own contracts.
Future extensions must state who holds data authority, how it is withdrawn,
which external inputs are checked, and what evidence establishes each boundary.
